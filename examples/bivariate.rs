#![feature(generic_arg_infer)]

use std::io::Write;

use neuramethyst::derivable::activation::Linear;
#[allow(unused_imports)]
use neuramethyst::derivable::activation::{LeakyRelu, Relu, Tanh};
use neuramethyst::derivable::loss::Euclidean;
use neuramethyst::derivable::regularize::NeuraElastic;
use neuramethyst::prelude::*;

use rand::Rng;

fn main() {
    let mut network = neura_network![
        neura_layer!("dense", 2, 8; LeakyRelu(0.01)),
        neura_layer!("dropout", 0.1),
        neura_layer!("dense", 8; LeakyRelu(0.01), NeuraElastic::new(0.0001, 0.002)),
        neura_layer!("dropout", 0.3),
        neura_layer!("dense", 8; LeakyRelu(0.01), NeuraElastic::new(0.0001, 0.002)),
        neura_layer!("dropout", 0.1),
        neura_layer!("dense", 4; LeakyRelu(0.1), NeuraElastic::new(0.0001, 0.002)),
        neura_layer!("dense", 2; Linear),
        neura_layer!("softmax"),
    ];
    // println!("{:#?}", network);

    let mut rng = rand::thread_rng();
    let inputs = (0..=1).cycle().map(move |category| {
        let (x, y) = if category == 0 {
            let radius: f64 = rng.gen_range(0.0..1.0);
            let radius = radius.sqrt();
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        } else {
            let radius: f64 = rng.gen_range(1.0..2.0);
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            (angle.cos() * radius, angle.sin() * radius)
        };

        ([x, y], one_hot::<2>(category))
    });

    let test_inputs: Vec<_> = inputs.clone().take(100).collect();

    let mut trainer = NeuraBatchedTrainer::new(0.25, 1000);
    trainer.log_epochs = 50;
    trainer.learning_momentum = 0.05;
    trainer.batch_size = 2000;

    trainer.train(
        NeuraBackprop::new(Euclidean),
        &mut network,
        inputs,
        &test_inputs,
    );

    let mut file = std::fs::File::create("target/bivariate.csv").unwrap();
    for (input, _target) in test_inputs {
        let guess = argmax(&network.eval(&input));
        writeln!(&mut file, "{},{},{}", input[0], input[1], guess).unwrap();
        // println!("{:?}", network.eval(&input));
    }

    // println!("{:#?}", network);
}

fn one_hot<const N: usize>(value: usize) -> [f64; N] {
    let mut res = [0.0; N];
    if value < N {
        res[value] = 1.0;
    }
    res
}

fn argmax(array: &[f64]) -> usize {
    let mut res = 0;

    for n in 1..array.len() {
        if array[n] > array[res] {
            res = n;
        }
    }

    res
}
