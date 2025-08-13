use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::Value;
use std::fmt;
use std::io::{Cursor, Read};

use crate::types::StreamingImageAnnotation;

/// A custom deserializer for ImageAnnotation that streams the image_data field
pub fn streaming_deserialize_image_annotation<'de, R>(
    deserializer: R,
) -> Result<StreamingImageAnnotation, R::Error>
where
    R: Deserializer<'de>,
{
    struct ImageAnnotationVisitor;

    impl<'de> Visitor<'de> for ImageAnnotationVisitor {
        type Value = StreamingImageAnnotation;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("struct ImageAnnotation")
        }

        fn visit_map<V>(self, mut map: V) -> Result<StreamingImageAnnotation, V::Error>
        where
            V: MapAccess<'de>,
        {
            let mut version = None;
            let mut flags = None;
            let mut shapes = None;
            let mut image_path = None;
            let mut image_data = None;
            let mut image_height = None;
            let mut image_width = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "version" => {
                        version = Some(map.next_value()?);
                    }
                    "flags" => {
                        flags = Some(map.next_value()?);
                    }
                    "shapes" => {
                        shapes = Some(map.next_value()?);
                    }
                    "imagePath" => {
                        image_path = Some(map.next_value()?);
                    }
                    "imageData" => {
                        // For image_data, we want to capture it as a raw string value
                        // without parsing it as a full string in memory
                        let value: Value = map.next_value()?;
                        if let Some(data_str) = value.as_str() {
                            // Create a streaming reader from the string
                            image_data =
                                Some(Box::new(Cursor::new(data_str.to_string()))
                                    as Box<dyn Read + Send>);
                        }
                    }
                    "imageHeight" => {
                        image_height = Some(map.next_value()?);
                    }
                    "imageWidth" => {
                        image_width = Some(map.next_value()?);
                    }
                    _ => {
                        // Skip unknown fields
                        map.next_value::<de::IgnoredAny>()?;
                    }
                }
            }

            let version = version.ok_or_else(|| de::Error::missing_field("version"))?;
            let image_path = image_path.ok_or_else(|| de::Error::missing_field("imagePath"))?;
            let image_height =
                image_height.ok_or_else(|| de::Error::missing_field("imageHeight"))?;
            let image_width = image_width.ok_or_else(|| de::Error::missing_field("imageWidth"))?;

            Ok(StreamingImageAnnotation {
                version,
                flags,
                shapes: shapes.unwrap_or_default(),
                image_path,
                image_data_stream: image_data,
                image_height,
                image_width,
            })
        }
    }

    const FIELDS: &[&str] = &[
        "version",
        "flags",
        "shapes",
        "imagePath",
        "imageData",
        "imageHeight",
        "imageWidth",
    ];
    deserializer.deserialize_struct("ImageAnnotation", FIELDS, ImageAnnotationVisitor)
}

/// Parse a JSON file with streaming support for image_data
pub fn read_and_parse_json_streaming<R: Read>(
    reader: R,
) -> Result<StreamingImageAnnotation, Box<dyn std::error::Error>> {
    let deserializer = &mut serde_json::Deserializer::from_reader(reader);
    streaming_deserialize_image_annotation(deserializer)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}
