mod images;

use clap::Parser;
use color_eyre::Result;
use faer::linalg::solvers::DenseSolveCore;
use faer_ext::{IntoFaer, IntoNdarray};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array1, Array2, Array3};
use std::{fmt::Debug, path::PathBuf, time::Instant};

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
    #[arg(short, long, value_parser = parse_nums::<f32, 4>)]
    matrix: [f32; 4],

    /// The amount to offset the image by (X,Y)
    #[arg(short = 'f', long, value_parser = parse_nums::<isize, 2>)]
    offset: Option<[isize; 2]>,

    /// Whether to apply the inverse transformation
    #[arg(short = 'n', long)]
    inverse: bool,

    /// The dimensions of the output image (set to 0 to keep the original dimensions)
    #[arg(short, long, value_parser = parse_nums::<usize, 2>)]
    dims: Option<[usize; 2]>,

    /// The color of the background in RGBA format
    #[arg(short, long, value_parser = parse_nums::<u8, 4>)]
    background: Option<[u8; 4]>,
}

fn parse_nums<T, const N: usize>(s: &str) -> Result<[T; N], String>
where
    T: std::str::FromStr + Clone + Debug,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let values: Vec<T> = s
        .split(',')
        .map(|v| v.trim().parse::<T>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    if values.len() != N {
        return Err(format!(
            "Expected {} elements, got {} ({:?})",
            N,
            values.len(),
            values
        ));
    }

    values.try_into().map_err(|e| format!("{:?}", e))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    println!("Loading image: {}...", args.input.display());
    let array = images::load_image(&args.input)?;
    let (height, width, _) = array.dim();
    println!("Input image dimensions: {}x{}", width, height);

    let matrix_vec = args.matrix.to_vec();
    let swapped_matrix = [matrix_vec[0], matrix_vec[2], matrix_vec[1], matrix_vec[3]];

    let matrix = {
        let mut matrix = Array2::from_shape_vec((2, 2), swapped_matrix.to_vec()).unwrap();
        if args.inverse {
            matrix = invert_matrix(matrix);
        }
        matrix
    };

    print_matrix(&matrix);

    if let Some(offset) = &args.offset {
        println!("Offset: ({}, {})", offset[0], offset[1]);
    }

    let offset = args.offset.unwrap_or([0, 0]);

    let out_dims = args.dims.unwrap_or([width, height]);
    let out_width = match out_dims[0] {
        0 => width,
        _ => out_dims[0],
    };
    let out_height = match out_dims[1] {
        0 => height,
        _ => out_dims[1],
    };

    println!("Output image dimensions: {}x{}", out_width, out_height);

    let mut output = Array3::<u8>::zeros((out_height, out_width, 4));

    if let Some(background) = args.background {
        let background: Array1<u8> = Array1::from_vec(background.to_vec());
        for y in 0..out_height {
            for x in 0..out_width {
                output.slice_mut(s![y, x, ..]).assign(&background);
            }
        }
    }

    let time = Instant::now();

    let mut min_x: isize = isize::MAX;
    let mut max_x: isize = isize::MIN;
    let mut min_y: isize = isize::MAX;
    let mut max_y: isize = isize::MIN;
    let mut cut_off: bool = false;

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
                let new_y = transformed[[1, 0]].round() as isize + offset[1];

                min_x = min_x.min(new_x);
                max_x = max_x.max(new_x);
                min_y = min_y.min(new_y);
                max_y = max_y.max(new_y);

                let new_y = out_height as isize - new_y - 1;

                if new_x >= 0
                    && new_x < out_width as isize
                    && new_y >= 0
                    && new_y < out_height as isize
                {
                    output
                        .slice_mut(s![new_y as usize, new_x as usize, ..])
                        .assign(&array.slice(s![y, x, ..]));
                } else {
                    cut_off = true;
                }

                pb.inc(1);
            }
        }

        pb.finish();
    }

    println!("Done! Took: {:?}", time.elapsed());

    println!(
        "Actual bounding box: ({}, {}) - ({}, {})",
        min_x, min_y, max_x, max_y
    );

    if cut_off {
        println!("Some pixels were cut off!");
    }

    images::save_image(output, &args.output)?;

    println!("Saved image: {}", args.output.display());

    Ok(())
}

fn print_matrix(matrix: &Array2<f32>) {
    println!("Transformation matrix:");
    for row in matrix.rows() {
        print!("| ");
        for value in row {
            print!("{:>5.2} ", value);
        }
        println!("|");
    }

    {
        let m_faer = matrix.view().into_faer();
        let det = m_faer.determinant();
        println!("Determinant: {}", det);
    }
}

fn invert_matrix(matrix: Array2<f32>) -> Array2<f32> {
    println!("Inverting matrix...");
    let m_faer = matrix.view().into_faer();
    let inv = m_faer.full_piv_lu().inverse();
    inv.as_ref().into_ndarray().to_owned()
}
