use std::f64::consts::PI;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::str::FromStr;

// import all the libraries
use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command, ValueHint};
#[cfg(feature = "png")]
use image::codecs::png;
use image::codecs::{bmp, pnm};
use image::{DynamicImage, ImageBuffer, Rgba};
// import our library (src/lib.rs)
use strange_attractor_renderer::config::{BrighnessConstants, Colors, Config, RenderKind};
use strange_attractor_renderer::{
    self, colorize, render, render_parallel, Attractor, ColorTransform, ParallelRenderer, Runtime,
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
/// Write image from user input.
/// Split out from main to allow both single-threaded and multithreaded
fn write_image_matches(
    image: ImageBuffer<Rgba<u16>, Vec<u16>>,
    matches: &ArgMatches,
    mut name: PathBuf,
    silent: bool,
) {
    let image = DynamicImage::ImageRgba16(image);

    // convert image to format.
    if !silent {
        println!("Converting image format.");
    }
    let image = match (matches.get_flag("transparent"), matches.get_flag("8bit")) {
        (true, false) => image,
        (false, false) => image.to_rgb16().into(),
        (true, true) => image.to_rgba8().into(),
        (false, true) => image.to_rgb8().into(),
    };

    if !silent {
        println!("Rendering complete. Writing file.");
    }
    let f = || {
        // writing file, depending on extension.
        if matches.get_flag("pam") {
            name.set_extension("pam");
            let mut file = file(&name);

            let codec = pnm::PnmEncoder::new(&mut file).with_subtype(pnm::PnmSubtype::ArbitraryMap);
            write_image(codec, image);
            return;
        } else if matches.get_flag("bmp") {
            name.set_extension("bmp");
            let mut file = file(&name);

            let encoder = bmp::BmpEncoder::new(&mut file);
            write_image(encoder, image);
            return;
        }
        #[cfg(feature = "png")]
        {
            name.set_extension("png");
            let mut file = file(&name);

            let codec = png::PngEncoder::new_with_quality(
                &mut file,
                png::CompressionType::Default,
                png::FilterType::Adaptive,
            );
            write_image(codec, image);
        }
        #[cfg(not(feature = "png"))]
        {
            eprintln!("Please specify an image format.");
            exit(1);
        }
    };
    f();

    println!("Wrote image to '{}'.", name.display());
}
/// Open a writeable file at `path`. Panics at any errors.
fn file(path: impl AsRef<Path>) -> impl Write {
    std::io::BufWriter::new(std::fs::File::create(path).unwrap())
}

/// Iterate over sequence angles
struct AngleIter {
    end: f64,
    curr: f64,
    step: f64,
    file: PathBuf,
    iter: usize,
    needed_digits: usize,
}
impl AngleIter {
    fn new(start: f64, end: f64, step: f64, file: PathBuf) -> Self {
        // estimate iterations, to get the needed digits to represent the sequence.
        let count = (end - start - step / 2.) / step;
        let needed_digits = if count as usize <= 1 {
            0
        } else {
            count.log10().ceil() as usize
        };

        Self {
            end,
            curr: start,
            step,
            file,
            iter: 0,
            needed_digits,
        }
    }
}
impl Iterator for AngleIter {
    type Item = (f64, PathBuf);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr + self.step / 2. < self.end {
            let v = self.curr;
            self.curr += self.step;

            // construct the new file name
            let mut file_name = self
                .file
                .file_stem()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("attractor")
                .to_owned();
            if self.needed_digits > 0 {
                // left aligned `digits` number of zeroes
                file_name.push_str(&format!(
                    "{:0>digits$}",
                    self.iter,
                    digits = self.needed_digits,
                ));
            }
            let mut file_name = PathBuf::from(file_name);
            if let Some(ext) = self.file.extension() {
                file_name.set_extension(ext);
            }

            let file = self.file.with_file_name(file_name);

            self.iter += 1;
            // Convert to radians
            let v = v * PI / 180.;

            Some((v, file))
        } else if self.iter == 0 {
            self.iter += 1;
            Some((self.curr, self.file.clone()))
        } else {
            None
        }
    }
}

fn main() {
    // split these up, which makes the formatting work on the long builder below.
    // This is probably a rustfmt bug.
    let jobs_per_thread_help = "Number of pieces to split the rendering up in per thread. This enables other faster threads to pick up the slack. Set to 1 to disable.";
    let brighness_help =
        "Offset the brightness. You generally want to decrease this if you have > 1e8 iterations.";

    let mut command = Command::new("strange-attractor-renderer")
        .arg(
            Arg::new("depth")
                .long("depth")
                .action(ArgAction::SetTrue)
                .help("output depth information"),
        )
        .arg(
            Arg::new("8bit")
                .long("8-bit")
                .short('8')
                .action(ArgAction::SetTrue)
                .help("Write image in an 8-bit format"),
        )
        .arg(
            Arg::new("transparent")
                .long("transparent")
                .short('t')
                .action(ArgAction::SetTrue)
                .help("Add transparency to the image"),
        )
        .arg(
            Arg::new("iterations")
                .long("iterations")
                .short('i')
                .value_hint(ValueHint::Other)
                .help("Number of iterations")
                .value_parser(parse_validate::<usize>)
                .default_value("10000000"),
        )
        .arg(
            Arg::new("width")
                .long("width")
                .short('w')
                .value_hint(ValueHint::Other)
                .help("Width of image")
                .value_parser(parse_validate::<u32>)
                .default_value("1920"),
        )
        .arg(
            Arg::new("height")
                .long("height")
                .short('v')
                .value_hint(ValueHint::Other)
                .help("Height of image")
                .value_parser(parse_validate::<u32>)
                .default_value("1080"),
        )
        .arg(
            Arg::new("preset")
                .long("preset")
                .short('p')
                .help("Which built-in attractor to render")
                .value_parser(clap::builder::PossibleValuesParser::new([
                    "poisson-saturne",
                    "solar-sail",
                ]))
                .default_value("poisson-saturne"),
        )
        .group(
            ArgGroup::new("format")
                .arg("pam")
                .arg("bmp")
                .requires("8bit"),
        )
        .arg(
            Arg::new("pam")
                .long("pam")
                .action(ArgAction::SetTrue)
                .help("Use PAM format, a bitmap-like format. 16-bit images are not supported.")
                .alias("pnm")
                .alias("pbm"),
        )
        .arg(
            Arg::new("bmp")
                .long("bmp")
                .action(ArgAction::SetTrue)
                .help("Use BMP format. 16-bit images are not supported.")
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
                .action(ArgAction::SetTrue)
                .help("Run on single thread"),
        )
        .arg(
            Arg::new("silent")
                .long("silent")
                .short('q')
                .action(ArgAction::SetTrue)
                .help("Decrease verbosity"),
        )
        .arg(
            Arg::new("jobs_per_thread")
                .long("jobs-per-thread")
                .short('j')
                .conflicts_with("singlethread")
                .help(jobs_per_thread_help)
                .value_parser(parse_validate::<NonZeroUsize>)
                .value_hint(ValueHint::Other)
                .default_value("12"),
        )
        .arg(
            Arg::new("angle")
                .long("angle")
                .short('a')
                .help("Angle to view attractor from (degrees)")
                .value_hint(ValueHint::Other)
                .value_parser(parse_validate::<f64>)
                .allow_hyphen_values(true)
                .default_value("0"),
        )
        .arg(
            Arg::new("brightness_offset")
                .long("brightness-offset")
                .short('b')
                .help(brighness_help)
                .default_value("-0.15")
                .value_hint(ValueHint::Other)
                .allow_hyphen_values(true)
                .value_parser(parse_validate::<f64>),
        )
        .subcommand(
            Command::new("sequence")
                .about(
                    "Render a sequence of frames rotating around the attractor.\n\
            All the arguments passed before this subcommand is used when crating the images.",
                )
                .arg(
                    Arg::new("start")
                        .long("start")
                        .short('s')
                        .help("The angle to start the animation from (degrees)")
                        .value_hint(ValueHint::Other)
                        .default_value("0"),
                )
                .arg(
                    Arg::new("end")
                        .long("end")
                        .short('e')
                        .help("The angle to end the animation at (degrees)")
                        .value_hint(ValueHint::Other)
                        .value_parser(parse_validate::<f64>)
                        .default_value("360"),
                )
                .arg(
                    Arg::new("step")
                        .long("step")
                        .short('d')
                        .help("Amount to change the angle for each frame (degrees)")
                        .value_hint(ValueHint::Other)
                        .value_parser(|v: &str| {
                            let v = parse_validate::<f64>(v)?;
                            if v <= 0. {
                                Err("step must be a positive".to_owned())
                            } else {
                                Ok(())
                            }
                        })
                        .default_value("0.5"),
                ),
        );

    // shell completion things
    #[cfg(feature = "complete")]
    {
        command = clap_autocomplete::add_subcommand(command);
    }

    let sequence_invalid = command.error(
        clap::error::ErrorKind::InvalidValue,
        "sequence end must be after start",
    );

    // shell completion things
    #[cfg(feature = "complete")]
    let command_copy = command.clone();

    let matches = command.get_matches();

    // shell completion things
    #[cfg(feature = "complete")]
    match clap_autocomplete::test_subcommand(&matches, command_copy) {
        Some(Ok(())) => {
            exit(1);
        }
        Some(Err(err)) => {
            eprintln!("Insufficient permissions, consider using --print flag: {err}");
            exit(1);
        }
        None => {}
    }

    // built-in attractor to use as the "base"
    match matches
        .get_one::<String>("preset")
        .expect("we have provided a default value")
        .as_str()
    {
        "poisson-saturne" => run(Config::poisson_saturne(), &matches, sequence_invalid),
        "solar-sail" => run(Config::solar_sail(), &matches, sequence_invalid),
        _ => unreachable!("clap validation should not allow any other values. Please report bug."),
    };
}

fn run(
    inherit: Config<impl Attractor + Send + Sync + 'static, impl ColorTransform>,
    matches: &ArgMatches,
    sequence_invalid: clap::Error,
) {
    // construct config
    let mut config = Config {
        iterations: *matches.get_one("iterations").unwrap(),
        width: *matches.get_one("width").unwrap(),
        height: *matches.get_one("height").unwrap(),

        transparent: matches.get_flag("transparent"),
        colors: Colors {
            brighness: BrighnessConstants {
                offset: *matches
                    .get_one("brightness_offset")
                    .expect("we have a default value and validated the input"),
                ..Default::default()
            },
            ..Default::default()
        },

        silent: matches.get_flag("silent"),

        ..inherit
    };
    config.render = if matches.get_flag("depth") {
        RenderKind::Depth
    } else {
        RenderKind::Gas
    };

    // get output file name
    let name = {
        let path = Path::new(
            matches
                .get_one::<String>("name")
                .expect("We provided a default value."),
        );
        let mut name = PathBuf::new();
        name.push(path.parent().unwrap_or_else(|| Path::new("/")));
        if let Some(stem) = path.file_stem() {
            name.push(stem);
        }
        name
    };

    let angle_iter = if let Some(matches) = matches.subcommand_matches("sequence") {
        let start = *matches
            .get_one("start")
            .expect("we have a default value and validated the input");
        let end = *matches
            .get_one("end")
            .expect("we have a default value and validated the input");
        let step = *matches
            .get_one("step")
            .expect("we have a default value and validated the input");
        if end <= start {
            sequence_invalid.exit();
        }
        AngleIter::new(start, end, step, name)
    } else {
        // get viewing angle
        let angle: f64 = *matches
            .get_one("angle")
            .expect("we have a default value and validated the input");
        AngleIter::new(angle, angle, 1., name)
    };

    // render image
    if matches.get_flag("singlethread") {
        let mut runtime = Runtime::new(&config);
        for (angle, name) in angle_iter {
            config.angle = angle;

            render(&config, &mut runtime);
            let image = colorize(&config, &runtime);
            write_image_matches(image, matches, name, config.silent);
            runtime.reset();
        }
    } else {
        let mut renderer = ParallelRenderer::new();
        let mut encoders = Vec::new();

        for (angle, name) in angle_iter {
            config.angle = angle;

            let image = render_parallel(
                &mut renderer,
                config.clone(),
                matches
                    .get_one::<NonZeroUsize>("jobs_per_thread")
                    .expect("we have a default value and validated the input")
                    .get(),
            );
            let matches = matches.clone();
            let handle = std::thread::spawn(move || {
                write_image_matches(image, &matches, name, config.silent);
            });
            encoders.push(handle);
        }

        encoders
            .into_iter()
            .for_each(|thread| thread.join().expect("encoder thread panicked"));
        renderer.shutdown();
    }
}
