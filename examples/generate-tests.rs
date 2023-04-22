#![feature(generic_arg_infer)]

use std::fs::File;

use nalgebra::dvector;

use neuramethyst::derivable::activation::{Relu, Tanh};
use neuramethyst::derivable::loss::Euclidean;
use neuramethyst::prelude::*;

fn main() {
    let mut network = neura_sequential![
        neura_layer!("dense", 4, f64).activation(Relu),
        neura_layer!("dense", 1, f64).activation(Tanh)
    ]
    .construct(NeuraShape::Vector(2))
    .unwrap();

    let inputs = [
        (dvector![0.0, 0.0], dvector![0.0]),
        (dvector![0.0, 1.0], dvector![1.0]),
        (dvector![1.0, 0.0], dvector![1.0]),
        (dvector![1.0, 1.0], dvector![0.0]),
    ];

    let mut trainer = NeuraBatchedTrainer::new(0.05, 1);
    trainer.batch_size = 1;

    let mut parameters = vec![(
        network.layer.weights.clone(),
        network.layer.bias.clone(),
        network.child_network.layer.weights.clone(),
        network.child_network.layer.bias.clone(),
    )];

    for iteration in 0..4 {
        trainer.train(
            &NeuraBackprop::new(Euclidean),
            &mut network,
            inputs.iter().cloned().skip(iteration).take(1),
            &inputs,
        );

        parameters.push((
            network.layer.weights.clone(),
            network.layer.bias.clone(),
            network.child_network.layer.weights.clone(),
            network.child_network.layer.bias.clone(),
        ));
    }

    let mut output = File::create("tests/xor.json").unwrap();
    serde_json::to_writer(&mut output, &parameters).unwrap();
}
