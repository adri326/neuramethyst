#![feature(generic_arg_infer)]

use nalgebra::dvector;

use neuramethyst::cycle_shuffling;
use neuramethyst::derivable::activation::Relu;
use neuramethyst::derivable::loss::Euclidean;
use neuramethyst::prelude::*;

fn main() {
    let mut network = neura_sequential![
        neura_layer!("dense", 4).activation(Relu),
        neura_layer!("dense", 3).activation(Relu),
        neura_layer!("dense", 1).activation(Relu)
    ]
    .construct(NeuraShape::Vector(2))
    .unwrap();

    let inputs = [
        (dvector![0.0, 0.0], dvector![0.0]),
        (dvector![0.0, 1.0], dvector![1.0]),
        (dvector![1.0, 0.0], dvector![1.0]),
        (dvector![1.0, 1.0], dvector![0.0]),
    ];

    for (input, target) in &inputs {
        println!(
            "Input: {:?}, target: {}, actual: {:.3}",
            &input,
            target[0],
            network.eval(&input)[0]
        );
    }

    let mut trainer = NeuraBatchedTrainer::new().learning_rate(0.05).iterations(0);
    trainer.batch_size = 6;
    trainer.log_iterations = 250;
    trainer.learning_momentum = 0.01;

    trainer.train(
        &NeuraBackprop::new(Euclidean),
        &mut network,
        cycle_shuffling(inputs.iter().cloned(), rand::thread_rng()),
        &inputs,
    );

    for (input, target) in inputs {
        println!(
            "Input: {:?}, target: {}, actual: {:.3}",
            &input,
            target[0],
            network.eval(&input)[0]
        );
    }
}
