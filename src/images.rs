use std::path::PathBuf;

use color_eyre::{eyre::ContextCompat, Result};
use image::{ImageReader, RgbImage};
use ndarray::Array3;

pub(crate) type ImageArray = ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 3]>>;

pub(crate) fn load_image(path: &PathBuf) -> Result<ImageArray> {
    let img = ImageReader::open(path)?.decode()?.into_rgb8();

    let (width, height) = (img.width() as usize, img.height() as usize);

    Ok(Array3::<u8>::from_shape_vec(
        (height, width, 3),
        img.as_raw().to_vec(),
    )?)
}

pub(crate) fn save_image(array: ImageArray, path: &PathBuf) -> Result<()> {
    let array = array.as_standard_layout().into_owned();

    let (height, width, _) = array.dim();

    let (flattened, _) = array.into_raw_vec_and_offset();

    let output_img = RgbImage::from_raw(width as u32, height as u32, flattened)
        .wrap_err("Failed to create image from array")?;

    output_img.save(path)?;

    Ok(())
}
