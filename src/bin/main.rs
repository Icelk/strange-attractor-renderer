use image::codecs::png;
use image::{ImageEncoder, EncodableLayout};
use strange_attractor_renderer::{self, render, Config, Runtime};

fn main() {
    let config = Config::default();
    let mut runtime = Runtime::new(&config);
    let image = render(&config, &mut runtime, 0.);

    let mut file = std::fs::File::create("test.png").unwrap();
    let codec = png::PngEncoder::new_with_quality(
        &mut file,
        png::CompressionType::Default,
        png::FilterType::Adaptive,
    );

    println!("Rendering complete. Writing file.");

    codec.write_image(
        image.as_bytes(),
        image.width(),
        image.height(),
        image::ColorType::Rgba16,
    ).unwrap();
}
