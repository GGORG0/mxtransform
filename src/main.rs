mod images;

use clap::Parser;
use color_eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array2, Array3, s};
use std::{path::PathBuf, time::Instant};

/// Transform images with the help of matrices
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The name of the input file
    #[arg(short, long)]
    input: PathBuf,

    /// The name of the output file
    #[arg(short, long)]
    output: PathBuf,

    /// The transformation matrix to apply to the image (Xx,Xy,Yx,Yy)
    #[arg(short, long, value_parser = parse_matrix)]
    matrix: [f32; 4],

    /// The amount to offset the image by (X,Y)
    #[arg(short = 'f', long, value_parser = parse_offset)]
    offset: Option<[isize; 2]>,
}

fn parse_matrix(s: &str) -> Result<[f32; 4], String> {
    let values: Vec<f32> = s
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if values.len() != 4 {
        return Err(format!("Expected 4 elements, got {}", values.len()));
    }

    Ok([values[0], values[2], values[1], values[3]])
}

fn parse_offset(s: &str) -> Result<[isize; 2], String> {
    let values: Vec<isize> = s
        .split(',')
        .map(|v| v.trim().parse::<isize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if values.len() != 2 {
        return Err(format!("Expected 2 elements, got {}", values.len()));
    }

    Ok([values[0], values[1]])
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    println!("Loading image: {}...", args.input.display());
    let array = images::load_image(&args.input)?;
    let (height, width, _) = array.dim();
    println!("Image dimensions: {}x{}", width, height);

    let matrix = Array2::from_shape_vec((2, 2), args.matrix.to_vec()).unwrap();

    println!("Transformation matrix:");
    for row in matrix.rows() {
        print!("| ");
        for value in row {
            print!("{:>5.2} ", value);
        }
        println!("|");
    }

    if let Some(offset) = &args.offset {
        println!("Offset: ({}, {})", offset[0], offset[1]);
    }

    let offset = args.offset.unwrap_or([0, 0]);

    let mut output = Array3::<u8>::zeros((height, width, 3));

    let time = Instant::now();

    {
        let pb = ProgressBar::new((height * width) as u64);
        pb.set_style(
            ProgressStyle::with_template("{wide_bar} {percent_precise}% ({eta})").unwrap(),
        );

        for y in 0..height {
            for x in 0..width {
                let pos = Array2::from_shape_vec((2, 1), vec![x as f32, (height - y - 1) as f32])
                    .unwrap();
                let transformed = matrix.dot(&pos);

                let new_x = transformed[[0, 0]].round() as isize + offset[0];
                let new_y = height as isize - transformed[[1, 0]].round() as isize - 1 - offset[1];

                if new_x >= 0 && new_x < width as isize && new_y >= 0 && new_y < height as isize {
                    output
                        .slice_mut(s![new_y as usize, new_x as usize, ..])
                        .assign(&array.slice(s![y, x, ..]));
                }

                pb.inc(1);
            }
        }

        pb.finish();
    }

    println!("Done! Took: {:?}", time.elapsed());

    images::save_image(output, &args.output)?;

    println!("Saved image: {}", args.output.display());

    Ok(())
}
