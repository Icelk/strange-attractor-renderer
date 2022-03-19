use std::f64::consts::PI;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::str::FromStr;

use clap::{Arg, ArgGroup, Command, ValueHint};
use image::codecs::{bmp, png, pnm};
use image::DynamicImage;
use strange_attractor_renderer::render_parallel;
use strange_attractor_renderer::{
    self, colorize, config::Config, config::RenderKind, render, Runtime,
};

/// validate that a argument passed by the user is valid, according to the parsing of type `T`.
fn parse_validate<T: FromStr>(s: &str) -> Result<T, String>
where
    <T as FromStr>::Err: ToString,
{
    s.parse().map_err(|e: T::Err| e.to_string())
}
/// helper to write file using a generic encoder (e.g. PNG, BMP)
fn write_image(encoder: impl image::ImageEncoder, image: DynamicImage) {
    encoder
        .write_image(
            image.as_bytes(),
            image.width(),
            image.height(),
            image.color(),
        )
        .unwrap();
}
/// Open a writeable file at `path`. Panics at any errors.
fn file(path: impl AsRef<Path>) -> impl Write {
    std::io::BufWriter::new(std::fs::File::create(path).unwrap())
}

fn main() {
    let mut command = Command::new("strange-attractor-renderer")
        .arg(
            Arg::new("depth")
                .long("depth")
                .help("output depth information"),
        )
        .arg(
            Arg::new("8bit")
                .long("8-bit")
                .short('8')
                .help("Write image in an 8-bit format"),
        )
        .arg(
            Arg::new("transparent")
                .long("transparent")
                .short('t')
                .help("output transparency"),
        )
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .short('i')
                .value_hint(ValueHint::Other)
                .help("Number of iterations")
                .validator(parse_validate::<usize>)
                .default_value("10000000"),
        )
        .arg(
            Arg::new("width")
                .long("width")
                .short('w')
                .value_hint(ValueHint::Other)
                .help("Width of image")
                .validator(parse_validate::<u32>)
                .default_value("1920"),
        )
        .arg(
            Arg::new("height")
                .long("height")
                .short('h')
                .value_hint(ValueHint::Other)
                .help("Height of image")
                .validator(parse_validate::<u32>)
                .default_value("1080"),
        )
        .group(ArgGroup::new("format").arg("pam").arg("bmp"))
        .arg(
            Arg::new("pam")
                .long("pam")
                .help("Use PAM format, a bitmap-like format")
                .alias("pnm")
                .alias("pbm"),
        )
        .arg(
            Arg::new("bmp")
                .long("bmp")
                .help("Use BMP format")
                .alias("bitmap"),
        )
        .arg(
            Arg::new("name")
                .long("file-name")
                .short('o')
                .help("Write to file name")
                .value_hint(ValueHint::FilePath)
                .default_value("attractor"),
        )
        .arg(
            Arg::new("singlethread")
                .long("single-thread")
                .short('s')
                .help("Run on single thread"),
        )
        .arg(
            Arg::new("angle")
                .long("angle")
                .short('a')
                .help("Angle to view attractor from (degrees)")
                .value_hint(ValueHint::Other)
                .validator(parse_validate::<f64>)
                .default_value("0"),
        );

    // shell completion things
    #[cfg(feature = "complete")]
    {
        command = clap_autocomplete::add_subcommand(command);
    }

    let command_copy = command.clone();
    let matches = command.get_matches();

    // shell completion things
    #[cfg(feature = "complete")]
    match clap_autocomplete::test_subcommand(&matches, command_copy) {
        Some(Ok(())) => {
            exit(1);
        }
        Some(Err(err)) => {
            eprintln!("Insufficient permissions: {err}");
            exit(1);
        }
        None => {}
    }

    // get output file name
    let mut name = {
        let path = Path::new(
            matches
                .value_of("name")
                .expect("We provided a default value."),
        );
        let mut name = PathBuf::new();
        name.push(path.parent().unwrap_or_else(|| Path::new("/")));
        if let Some(stem) = path.file_stem() {
            name.push(stem);
        }
        name
    };

    // construct config
    let mut config = Config {
        iterations: matches.value_of_t("iterations").unwrap(),
        width: matches.value_of_t("width").unwrap(),
        height: matches.value_of_t("height").unwrap(),

        transparent: matches.is_present("transparent"),
        ..Default::default()
    };
    config.render = if matches.is_present("depth") {
        RenderKind::Depth
    } else {
        RenderKind::Gas
    };

    // check for restraints on formats
    if matches.is_present("pam") && !matches.is_present("8bit") {
        eprintln!("16-bit images not supported when using PAM format.");
        exit(1);
    }
    if matches.is_present("bmp") && !matches.is_present("8bit") {
        eprintln!("16-bit images not supported when using BMP format.");
        exit(1);
    }

    // get viewing angle
    let angle: f64 = matches
        .value_of_t("angle")
        .expect("we have a default value and validated the input");
    // Convert to radians
    let angle = angle * PI / 180.;

    // render image
    let image = if matches.is_present("singlethread") {
        let mut runtime = Runtime::new(&config);
        render(&config, &mut runtime, angle);
        colorize(&config, &runtime)
    } else {
        render_parallel(config.clone(), angle * PI / 180., 12)
    };
    let image = DynamicImage::ImageRgba16(image);

    // convert image to format.
    println!("Converting image format.");
    let image = match (config.transparent, matches.is_present("8bit")) {
        (true, false) => image,
        (false, false) => image.to_rgb16().into(),
        (true, true) => image.to_rgba8().into(),
        (false, true) => image.to_rgb8().into(),
    };

    println!("Rendering complete. Writing file.");
    // writing file, depending on extension.
    if matches.is_present("pam") {
        name.set_extension("pam");
        let mut file = file(&name);

        let codec = pnm::PnmEncoder::new(&mut file).with_subtype(pnm::PnmSubtype::ArbitraryMap);
        write_image(codec, image);
    } else if matches.is_present("bmp") {
        name.set_extension("bmp");
        let mut file = file(&name);

        let encoder = bmp::BmpEncoder::new(&mut file);
        write_image(encoder, image);
    } else {
        name.set_extension("png");
        let mut file = file(&name);

        let codec = png::PngEncoder::new_with_quality(
            &mut file,
            png::CompressionType::Default,
            png::FilterType::Adaptive,
        );
        write_image(codec, image);
    }
}
