#![feature(generic_arg_infer)]

use std::io::Write;

#[allow(unused_imports)]
use neuramethyst::derivable::activation::{LeakyRelu, Linear, Relu, Tanh};
use neuramethyst::derivable::loss::CrossEntropy;
use neuramethyst::derivable::regularize::NeuraL1;
use neuramethyst::prelude::*;

use rand::Rng;

fn main() {
    let mut network = neura_network![
        neura_layer!("dense", 2, 8; Relu, NeuraL1(0.001)),
        neura_layer!("dropout", 0.25),
        neura_layer!("dense", 2; Linear, NeuraL1(0.001)),
        neura_layer!("softmax"),
    ];

    let inputs = (0..1).cycle().map(move |_| {
        let mut rng = rand::thread_rng(); // TODO: move out
        let category = rng.gen_bool(0.5) as usize;
        let (x, y) = if category == 0 {
            let radius: f64 = rng.gen_range(0.0..2.0);
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        } else {
            let radius: f64 = rng.gen_range(3.0..5.0);
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        };

        ([x, y], neuramethyst::one_hot::<2>(category))
    });

    let test_inputs: Vec<_> = inputs.clone().take(10).collect();

    if std::env::args().any(|arg| arg == "draw") {
        for epoch in 0..200 {
            let mut trainer = NeuraBatchedTrainer::new(0.03, 10);
            trainer.batch_size = 10;

            trainer.train(
                NeuraBackprop::new(CrossEntropy),
                &mut network,
                inputs.clone(),
                &test_inputs,
            );

            let network = network.clone();
            draw_neuron_activation(|input| network.eval(&input).into_iter().collect(), 6.0);
            println!("{}", epoch);

            std::thread::sleep(std::time::Duration::new(0, 50_000_000));
        }
    } else {
        let mut trainer = NeuraBatchedTrainer::new(0.03, 20 * 50);
        trainer.batch_size = 10;
        trainer.log_iterations = 20;

        trainer.train(
            NeuraBackprop::new(CrossEntropy),
            &mut network,
            inputs.clone(),
            &test_inputs,
        );

        // println!("{}", String::from("\n").repeat(64));
        // draw_neuron_activation(|input| network.eval(&input).into_iter().collect(), 6.0);
    }

    let mut file = std::fs::File::create("target/bivariate.csv").unwrap();
    for (input, _target) in test_inputs {
        let guess = neuramethyst::argmax(&network.eval(&input));
        writeln!(&mut file, "{},{},{}", input[0], input[1], guess).unwrap();
    }
}

// TODO: move this to the library?
fn draw_neuron_activation<F: Fn([f64; 2]) -> Vec<f64>>(callback: F, scale: f64) {
    use viuer::Config;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut image = image::RgbImage::new(WIDTH, HEIGHT);

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x * 3.0).exp())
    }

    for y in 0..HEIGHT {
        let y2 = 2.0 * y as f64 / HEIGHT as f64 - 1.0;
        for x in 0..WIDTH {
            let x2 = 2.0 * x as f64 / WIDTH as f64 - 1.0;
            let activation = callback([x2 * scale, y2 * scale]);
            let r = (sigmoid(activation.get(0).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let g = (sigmoid(activation.get(1).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;
            let b = (sigmoid(activation.get(2).copied().unwrap_or(-1.0)) * 255.0).floor() as u8;

            *image.get_pixel_mut(x, y) = image::Rgb([r, g, b]);
        }
    }

    let config = Config {
        use_kitty: false,
        // absolute_offset: false,
        ..Default::default()
    };

    viuer::print(&image::DynamicImage::ImageRgb8(image), &config).unwrap();
}
