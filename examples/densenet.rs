use std::io::Write;

use nalgebra::{dvector, DVector};
use neuramethyst::derivable::activation::Linear;
use neuramethyst::derivable::loss::CrossEntropy;
use neuramethyst::derivable::regularize::NeuraL1;
use neuramethyst::{plot_losses, prelude::*};

use rand::Rng;

fn main() {
    let mut network = neura_residual![
        <= 0, 2;
        neura_layer!("dense", 4).regularization(NeuraL1(0.001));
        neura_layer!("dropout", 0.25);
        neura_layer!("dense", 2)
            .activation(Linear)
            .regularization(NeuraL1(0.001));
        neura_layer!("softmax");
    ]
    .construct(NeuraShape::Vector(2))
    .unwrap();

    let inputs = (0..1).cycle().map(move |_| {
        let mut rng = rand::thread_rng();
        let category = rng.gen_bool(0.5) as usize;
        let (x, y) = if category == 0 {
            let radius: f32 = rng.gen_range(0.0..2.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        } else {
            let radius: f32 = rng.gen_range(3.0..5.0);
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        };

        (dvector![x, y], one_hot(category, 2))
    });

    let test_inputs: Vec<_> = inputs.clone().take(10).collect();

    if std::env::args().any(|arg| arg == "draw") {
        for epoch in 0..200 {
            let mut trainer = NeuraBatchedTrainer::new(0.03, 10);
            trainer.batch_size = 10;

            trainer.train(
                &NeuraBackprop::new(CrossEntropy),
                &mut network,
                inputs.clone(),
                &test_inputs,
            );

            neuramethyst::draw_neuron_activation(
                |input| {
                    let output = network.eval(&dvector![input[0], input[1]]);
                    let estimation = output[0] / (output[0] + output[1]);

                    let color = network.eval(&dvector![input[0], input[1]]);

                    (&color / color.map(|x| x * x).sum() * estimation)
                        .into_iter()
                        .map(|x| x.abs() as f32)
                        .collect::<Vec<_>>()
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
                &NeuraBackprop::new(CrossEntropy),
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

    let mut file = std::fs::File::create("target/bivariate.csv").unwrap();
    for (input, _target) in test_inputs {
        let guess = neuramethyst::argmax(network.eval(&input).as_slice());
        writeln!(&mut file, "{},{},{}", input[0], input[1], guess).unwrap();
    }
}

fn one_hot(value: usize, categories: usize) -> DVector<f32> {
    let mut res = DVector::from_element(categories, 0.0);
    if value < categories {
        res[value] = 1.0;
    }
    res
}
