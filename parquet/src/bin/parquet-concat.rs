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

//! Binary that concatenates the column data of one or more parquet files
//!
//! # Install
//!
//! `parquet-concat` can be installed using `cargo`:
//! ```
//! cargo install parquet --features=cli
//! ```
//! After this `parquet-concat` should be available:
//! ```
//! parquet-concat out.parquet a.parquet b.parquet
//! ```
//!
//! The binary can also be built from the source code and run as follows:
//! ```
//! cargo run --features=cli --bin parquet-concat out.parquet a.parquet b.parquet
//! ```
//!
//! Note: this does not currently support preserving the page index or bloom filters
//!

use clap::Parser;
use parquet::bloom_filter::Sbbf;
use parquet::column::writer::ColumnCloseResult;
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::{ColumnChunkMetaData, PageIndexPolicy, ParquetMetaDataReader};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::ChunkReader;
use parquet::file::writer::SerializedFileWriter;
use std::fs::File;
use std::sync::Arc;

#[derive(Debug, Parser)]
#[clap(author, version)]
/// Concatenates one or more parquet files
struct Args {
    /// Path to output
    output: String,

    /// Path to input files
    input: Vec<String>,
}

fn read_bloom_filter<R: ChunkReader>(column: &ColumnChunkMetaData, input: &R) -> Option<Sbbf> {
    Sbbf::read_from_column_chunk(column, input).ok().flatten()
}

impl Args {
    fn run(&self) -> Result<()> {
        if self.input.is_empty() {
            return Err(ParquetError::General(
                "Must provide at least one input file".into(),
            ));
        }

        let output = File::create(&self.output)?;

        let inputs = self
            .input
            .iter()
            .map(|x| {
                let reader = File::open(x)?;
                // Enable reading page indexes if present
                let metadata = ParquetMetaDataReader::new()
                    .with_page_index_policy(PageIndexPolicy::Optional)
                    .parse_and_finish(&reader)?;
                Ok((reader, metadata))
            })
            .collect::<Result<Vec<_>>>()?;

        let expected = inputs[0].1.file_metadata().schema();
        for (_, metadata) in inputs.iter().skip(1) {
            let actual = metadata.file_metadata().schema();
            if expected != actual {
                return Err(ParquetError::General(format!(
                    "inputs must have the same schema, {expected:#?} vs {actual:#?}"
                )));
            }
        }

        let props = Arc::new(WriterProperties::builder().build());
        let schema = inputs[0].1.file_metadata().schema_descr().root_schema_ptr();
        let mut writer = SerializedFileWriter::new(output, schema, props)?;

        for (input, metadata) in inputs {
            let column_indexes = metadata.column_index();
            let offset_indexes = metadata.offset_index();

            for (rg_idx, rg) in metadata.row_groups().iter().enumerate() {
                let rg_column_indexes = column_indexes.and_then(|ci| ci.get(rg_idx));
                let rg_offset_indexes = offset_indexes.and_then(|oi| oi.get(rg_idx));
                let mut rg_out = writer.next_row_group()?;
                for (col_idx, column) in rg.columns().iter().enumerate() {
                    let bloom_filter = read_bloom_filter(column, &input);
                    let column_index = rg_column_indexes.and_then(|row| row.get(col_idx)).cloned();

                    let offset_index = rg_offset_indexes.and_then(|row| row.get(col_idx)).cloned();

                    let result = ColumnCloseResult {
                        bytes_written: column.compressed_size() as _,
                        rows_written: rg.num_rows() as _,
                        metadata: column.clone(),
                        bloom_filter,
                        column_index,
                        offset_index,
                    };
                    rg_out.append_column(&input, result)?;
                }
                rg_out.close()?;
            }
        }

        writer.close()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    Args::parse().run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, LargeStringArray, RecordBatch, RecordBatchReader};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
    use std::clone::Clone;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    #[should_panic]
    fn test_0_cli_arguments_is_invalid() {
        Args::try_parse_from(vec![""]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_single_cli_argument_is_invalid() {
        Args::try_parse_from(vec!["parquet-concat"]).unwrap();
    }

    #[test]
    fn test_need_at_least_one_input_file() -> Result<()> {
        let tmpf = TempDir::new().unwrap();
        let out_name = tmpf
            .path()
            .join("out.parquet")
            .into_os_string()
            .into_string()
            .unwrap();
        let args = Args::try_parse_from(vec!["parquet-concat", &out_name]).unwrap();

        match args.run() {
            Err(ParquetError::General(msg)) if msg == "Must provide at least one input file" => {}
            _ => panic!("expected General ParquetError"),
        }
        Ok(())
    }

    #[test]
    fn test_input_files_do_not_exist() -> Result<()> {
        let tmpf = TempDir::new().unwrap();
        let out_name = tmpf.path().join("out.parquet");

        let in1_name = tmpf.path().join("in1.parquet");
        let in2_name = tmpf.path().join("in2.parquet");
        let in3_name = tmpf.path().join("in3.parquet");

        let inputs = vec![
            in1_name.to_str().unwrap().to_string(),
            in2_name.to_str().unwrap().to_string(),
            in3_name.to_str().unwrap().to_string(),
        ];

        let args = Args {
            output: out_name.to_str().unwrap().to_string(),
            input: inputs,
        };

        let o = args.run();

        match o {
            Err(ParquetError::External(_)) => (),
            _ => panic!("expected External ParquetError"),
        }

        Ok(())
    }

    #[test]
    fn test_concatenate_3_files() -> Result<()> {
        let num_files = 3;

        let tmpf = TempDir::new().unwrap();
        let res = run_concat_n_files(&tmpf, num_files, DatasetOne::to_rcb);

        match res {
            Ok(()) => (),
            Err(e) => panic!("expected Ok(()), got {}", e),
        }

        let reader = File::open(tmpf.path().join("out.parquet"))?;
        let metadata = ParquetMetaDataReader::new().parse_and_finish(&reader)?;
        let filemeta = metadata.file_metadata();

        assert_eq!(
            filemeta.num_rows(),
            (num_files as i64) * DatasetOne::num_rows()
        );

        let mut expected = Vec::<DatasetOne>::with_capacity(3 * 16);
        for ind in 0..num_files {
            expected.extend(DatasetOne::get());
        }

        let batch_reader = ParquetRecordBatchReader::try_new(reader, 1000000);

        let _batches: Vec<RecordBatch> = batch_reader.iter().map(|res| res.unwrap()).collect();

        Ok(())
    }

    #[test]
    fn test_concatenate_3000_files() -> Result<()> {
        let num_files = 3000;
        let tmpf = TempDir::new().unwrap();
        let res = run_concat_n_files(&tmpf, num_files, DatasetOne::to_rcb);
        match res {
            Err(ParquetError::External(_)) => (),
            _ => panic!("expected External ParquetError"),
        }
        Ok(())
    }

    #[test]
    fn test_concatenate_2_files_different_schemas() -> Result<()> {
        let tmpf = TempDir::new().unwrap();
        let out_name = tmpf.path().join("out.parquet");
        let inputs: Vec<String> = vec![
            tmpf.path().join("in1.parquet").display().to_string(),
            tmpf.path().join("in2.parquet").display().to_string(),
        ];

        DatasetOne::to_rcb(&inputs[0]);
        DatasetTwo::to_rcb(&inputs[1]);

        let args = Args {
            output: out_name.to_str().unwrap().to_string(),
            input: inputs,
        };

        match args.run() {
            Err(ParquetError::General(msg)) => {
                assert!(msg.starts_with("inputs must have the same schema"));
            }
            _ => panic!("expected ParquetError::General"),
        };

        Ok(())
    }

    /// First dataset has differing `length` column in i32
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct DatasetOne {
        pub length: i32,
        pub annotation: String,
    }

    impl DatasetOne {
        pub fn get_schema() -> Arc<Schema> {
            Arc::new(Schema::new(vec![
                Field::new("total_length", DataType::Int32, false),
                Field::new("animal", DataType::LargeUtf8, false),
            ]))
        }

        pub fn num_rows() -> i64 {
            16
        }

        pub fn get_recordbatch() -> RecordBatch {
            let schema = Self::get_schema();

            // TODO: let values be length of strings below
            let col1_data = Arc::new(Int32Array::from_iter_values([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            ])) as Arc<dyn arrow_array::Array>;
            let col2_data = Arc::new(LargeStringArray::from_iter_values([
                "mouse", "cat", "horse", "mouse", "cat", "horse", "mouse", "cat", "mouse", "cat",
                "horse", "mouse", "cat", "horse", "mouse", "cat",
            ])) as Arc<dyn arrow_array::Array>;

            assert_eq!(
                // defensive check
                col1_data.len(),
                col2_data.len(),
                "columns are of equal length"
            );

            RecordBatch::try_new(schema.clone(), vec![col1_data, col2_data]).unwrap()
        }

        fn to_rcb(path: &str) {
            let record_batch = Self::get_recordbatch();

            assert_eq!(
                record_batch.num_rows() as i64,
                Self::num_rows(),
                "defensive check",
            );

            let mut buffer = File::create(path).unwrap();
            let mut writer =
                ArrowWriter::try_new(&mut buffer, record_batch.schema(), None).unwrap();
            writer.write(&record_batch).unwrap();

            writer.close().unwrap();
        }

        fn from_recordbatch(record_batch: &RecordBatch) -> Vec<DatasetOne> {
            let column1 = record_batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("failed to downcast");
            let column2 = record_batch
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("failed to downcast");

            let mut ds: Vec<DatasetOne> = vec![];
            for (ii, ss) in column1.iter().zip(column2.iter()) {
                ds.push(DatasetOne {
                    length: ii.expect("to be number"),
                    annotation: ss.expect("to be string").to_string(),
                });
            }
            ds
        }

        fn get() -> Vec<DatasetOne> {
            let rb = Self::get_recordbatch();
            Self::from_recordbatch(&rb)
        }
    }

    /// First dataset has differing `length` column in i64
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct DatasetTwo {
        pub length: i64,
        pub annotation: String,
    }

    impl DatasetTwo {
        pub fn get_schema() -> Arc<Schema> {
            Arc::new(Schema::new(vec![
                Field::new("length", DataType::Int64, false),
                Field::new("String", DataType::LargeUtf8, false),
            ]))
        }

        pub fn get_recordbatch() -> RecordBatch {
            let schema = Self::get_schema();

            // TODO: let values be length of strings below
            let col1_data = Arc::new(Int64Array::from_iter_values([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            ])) as Arc<dyn arrow_array::Array>;
            let col2_data = Arc::new(LargeStringArray::from_iter_values([
                "mouse", "cat", "horse", "mouse", "cat", "horse", "mouse", "cat", "mouse", "cat",
                "horse", "mouse", "cat", "horse", "mouse", "cat",
            ])) as Arc<dyn arrow_array::Array>;

            assert_eq!(
                // defensive check
                col1_data.len(),
                col2_data.len(),
                "columns are of equal length"
            );

            RecordBatch::try_new(schema.clone(), vec![col1_data, col2_data]).unwrap()
        }

        pub fn num_rows() -> i64 {
            16
        }

        fn to_rcb(path: &str) {
            let record_batch = Self::get_recordbatch();

            assert_eq!(
                record_batch.num_rows() as i64,
                Self::num_rows(),
                "defensive check",
            );

            let mut buffer = File::create(path).unwrap();
            let mut writer =
                ArrowWriter::try_new(&mut buffer, record_batch.schema(), None).unwrap();
            writer.write(&record_batch).unwrap();

            writer.close().unwrap();
        }
    }

    /// Create `num_files` files of type `DatasetOne`
    /// and then run `parquet-concat` on them
    fn run_concat_n_files(tmpf: &TempDir, num_files: usize, to_rcb: impl Fn(&str)) -> Result<()> {
        let out_name = tmpf.path().join("out.parquet");

        let mut inputs = Vec::<String>::with_capacity(num_files);
        for file_index in 1..=num_files {
            let in_name = tmpf.path().join(format!("in{file_index}.parquet"));
            to_rcb(in_name.to_str().unwrap());
            inputs.push(in_name.display().to_string());
        }

        let args = Args {
            output: out_name.to_str().unwrap().to_string(),
            input: inputs,
        };

        args.run()
    }
}
