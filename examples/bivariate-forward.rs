#![feature(generic_arg_infer)]

use nalgebra::{dvector, DVector};
#[allow(unused_imports)]
use neuramethyst::derivable::activation::{LeakyRelu, Linear, Relu, Tanh};
use neuramethyst::derivable::regularize::NeuraL1;
use neuramethyst::gradient_solver::NeuraForwardForward;
use neuramethyst::{plot_losses, prelude::*};

use rand::Rng;

fn main() {
    let mut network = neura_sequential![
        neura_layer!("dense", 10).regularization(NeuraL1(0.001)),
        neura_layer!("dropout", 0.25),
        neura_layer!("normalize"),
        neura_layer!("dense", 6).regularization(NeuraL1(0.001)),
    ]
    .construct(NeuraShape::Vector(3))
    .unwrap();

    let inputs = (0..1).cycle().map(move |_| {
        let mut rng = rand::thread_rng();
        let category = rng.gen_bool(0.5);
        let good = rng.gen_bool(0.5);
        let (x, y) = if category {
            let radius: f32 = rng.gen_range(0.0..2.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        } else {
            let radius: f32 = rng.gen_range(3.0..5.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        };

        if good {
            (dvector![x, y, category as u8 as f32], true)
        } else {
            (dvector![x, y, 1.0 - category as u8 as f32], false)
        }
    });

    let test_inputs: Vec<_> = inputs.clone().filter(|(_, good)| *good).take(10).collect();
    let threshold = 0.25f32;

    if std::env::args().any(|arg| arg == "draw") {
        for epoch in 0..200 {
            let mut trainer = NeuraBatchedTrainer::new(0.03, 10);
            trainer.batch_size = 10;

            trainer.train(
                &NeuraForwardForward::new(Tanh, threshold as f64),
                &mut network,
                inputs.clone(),
                &test_inputs,
            );

            // let network = network.clone().trim_tail().trim_tail();
            draw_neuron_activation(
                |input| {
                    let cat0 = network.eval(&dvector![input[0] as f32, input[1] as f32, 0.0]);
                    let cat1 = network.eval(&dvector![input[0] as f32, input[1] as f32, 1.0]);

                    let cat0_good = cat0.map(|x| x * x).sum();
                    let cat1_good = cat1.map(|x| x * x).sum();
                    let estimation = cat1_good / (cat0_good + cat1_good);

                    let cat0_norm = cat0 / cat0_good.sqrt();
                    let mut cat0_rgb = DVector::from_element(3, 0.0);

                    for i in 0..cat0_norm.len() {
                        cat0_rgb[i % 3] += cat0_norm[i].abs();
                    }

                    (cat0_rgb * estimation)
                        .into_iter()
                        .map(|x| *x as f64)
                        .collect()
                },
                6.0,
            );
            println!("{}", epoch);

            std::thread::sleep(std::time::Duration::new(0, 50_000_000));
        }
    } else {
        let mut trainer = NeuraBatchedTrainer::new(0.03, 20 * 50);
        trainer.batch_size = 10;
        trainer.log_iterations = 20;

        plot_losses(
            trainer.train(
                &NeuraForwardForward::new(Tanh, threshold as f64),
                &mut network,
                inputs.clone(),
                &test_inputs,
            ),
            128,
            48,
        );

        // println!("{}", String::from("\n").repeat(64));
        // draw_neuron_activation(|input| network.eval(&input).into_iter().collect(), 6.0);
    }
}

// TODO: move this to the library?
fn draw_neuron_activation<F: Fn([f64; 2]) -> Vec<f64>>(callback: F, scale: f64) {
    use viuer::Config;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut image = image::RgbImage::new(WIDTH, HEIGHT);

    fn sigmoid(x: f64) -> f64 {
        0.1 + 0.9 * x.abs().powf(0.8)
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
        truecolor: true,
        // absolute_offset: false,
        ..Default::default()
    };

    viuer::print(&image::DynamicImage::ImageRgb8(image), &config).unwrap();
}
