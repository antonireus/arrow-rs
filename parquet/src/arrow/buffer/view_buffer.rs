// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::arrow::record_reader::buffer::ValuesBuffer;
use crate::errors::{ParquetError, Result};
use arrow_array::builder::make_view;
use arrow_array::{ArrayRef, BinaryViewArray, StringViewArray};
use arrow_buffer::{Buffer, NullBuffer, ScalarBuffer};
use arrow_data::ByteView;
use arrow_schema::DataType as ArrowType;
use std::borrow::Cow;
use std::sync::Arc;

/// The high bit of each of the 12 bytes a view can hold inline
const INLINE_HIGH_BITS: u128 = 0x8080_8080_8080_8080_8080_8080;

/// A buffer of view type byte arrays that can be converted into
/// `GenericByteViewArray`
///
/// Note this does not reuse `GenericByteViewBuilder` due to the need to call `pad_nulls`
/// and reuse the existing logic for Vec in the parquet crate
#[derive(Debug, Default)]
pub struct ViewBuffer {
    pub views: Vec<u128>,
    pub buffers: Vec<Buffer>,
}

impl ViewBuffer {
    /// Create a new ViewBuffer with capacity for the specified number of views
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            views: Vec::with_capacity(capacity),
            buffers: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    pub fn append_block(&mut self, block: Buffer) -> u32 {
        let block_id = self.buffers.len() as u32;
        self.buffers.push(block);
        block_id
    }

    /// Converts this into an [`ArrayRef`] with the provided `data_type` and `null_buffer`
    /// Returns true if the values from index `first` on start on UTF-8 character boundaries
    pub fn on_char_boundaries(&self, first: usize) -> bool {
        // The 4 bytes after the length hold the first bytes of any non-empty value
        let first_byte = |view: u128| (view >> 32) as u8 as i8;
        self.views[first..]
            .iter()
            .all(|&view| view as u32 == 0 || first_byte(view) >= -0x40)
    }

    /// Replaces invalid UTF-8 in the values from index `first` on with U+FFFD
    pub fn replace_invalid_utf8(&mut self, first: usize) -> Result<()> {
        let block_id = self.buffers.len() as u32;
        let mut block = Vec::new();
        for view in &mut self.views[first..] {
            let len = *view as u32 as usize;
            // A short value is stored in the view itself, so ASCII is a single mask test
            if len <= 12 && (*view >> 32) & (INLINE_HIGH_BITS >> (96 - 8 * len)) == 0 {
                continue;
            }
            let inline = view.to_le_bytes();
            let value = match len {
                0..=12 => &inline[4..4 + len],
                _ => {
                    let view = ByteView::from(*view);
                    let offset = view.offset as usize;
                    &self.buffers[view.buffer_index as usize][offset..offset + len]
                }
            };
            if let Cow::Owned(value) = String::from_utf8_lossy(value) {
                let offset = u32::try_from(block.len())
                    .map_err(|_| general_err!("index overflow decoding byte array"))?;
                if value.len() > 12 {
                    block.extend_from_slice(value.as_bytes());
                }
                *view = make_view(value.as_bytes(), block_id, offset);
            }
        }
        if !block.is_empty() {
            self.append_block(Buffer::from_vec(block));
        }
        Ok(())
    }

    pub fn into_array(self, null_buffer: Option<Buffer>, data_type: &ArrowType) -> ArrayRef {
        let len = self.views.len();
        let views = ScalarBuffer::from(self.views);
        let nulls = null_buffer.and_then(|b| NullBuffer::from_unsliced_buffer(b, len));
        let buffers = self.buffers.into();
        match data_type {
            ArrowType::Utf8View => {
                // Safety: views were created correctly, and checked that the data is utf8 when building the buffer
                unsafe { Arc::new(StringViewArray::new_unchecked(views, buffers, nulls)) }
            }
            ArrowType::BinaryView => {
                // Safety: views were created correctly
                unsafe { Arc::new(BinaryViewArray::new_unchecked(views, buffers, nulls)) }
            }
            _ => panic!("Unsupported data type: {data_type}"),
        }
    }
}

impl ValuesBuffer for ViewBuffer {
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity)
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.views.reserve_exact(additional);
    }

    fn pad_nulls(
        &mut self,
        read_offset: usize,
        values_read: usize,
        levels_read: usize,
        valid_mask: &[u8],
    ) -> Result<()> {
        self.views
            .pad_nulls(read_offset, values_read, levels_read, valid_mask)
    }
}

#[cfg(test)]
mod tests {

    use arrow::array::make_view;
    use arrow_array::Array;

    use super::*;

    #[test]
    fn test_view_buffer_empty() {
        let buffer = ViewBuffer::with_capacity(0);
        let array = buffer.into_array(None, &ArrowType::Utf8View);
        let strings = array
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        assert_eq!(strings.len(), 0);
    }

    #[test]
    fn test_view_buffer_append_view() {
        let mut buffer = ViewBuffer::with_capacity(0);
        let data = b"0123456789long string to test string view";
        let string_buffer = Buffer::from(data);
        let block_id = buffer.append_block(string_buffer);

        buffer.views.push(make_view(&data[0..1], block_id, 0));
        buffer.views.push(make_view(&data[1..10], block_id, 1));
        buffer.views.push(make_view(&data[10..41], block_id, 10));

        let array = buffer.into_array(None, &ArrowType::Utf8View);
        let string_array = array
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();
        assert_eq!(
            string_array.iter().collect::<Vec<_>>(),
            vec![
                Some("0"),
                Some("123456789"),
                Some("long string to test string view"),
            ]
        );
    }

    #[test]
    fn test_view_buffer_pad_null() {
        let mut buffer = ViewBuffer::with_capacity(0);
        let data = b"0123456789long string to test string view";
        let string_buffer = Buffer::from(data);
        let block_id = buffer.append_block(string_buffer);

        buffer.views.push(make_view(&data[0..1], block_id, 0));
        buffer.views.push(make_view(&data[1..10], block_id, 1));
        buffer.views.push(make_view(&data[10..41], block_id, 10));

        let valid = [true, false, false, true, false, false, true];
        let valid_mask = Buffer::from_iter(valid.iter().copied());

        buffer
            .pad_nulls(1, 2, valid.len() - 1, valid_mask.as_slice())
            .unwrap();

        let array = buffer.into_array(Some(valid_mask), &ArrowType::Utf8View);
        let strings = array
            .as_any()
            .downcast_ref::<arrow::array::StringViewArray>()
            .unwrap();

        assert_eq!(
            strings.iter().collect::<Vec<_>>(),
            vec![
                Some("0"),
                None,
                None,
                Some("123456789"),
                None,
                None,
                Some("long string to test string view"),
            ]
        );
    }
}
