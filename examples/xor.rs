#![feature(generic_arg_infer)]

use neuramethyst::prelude::*;
use neuramethyst::derivable::activation::{Relu};
use neuramethyst::derivable::loss::Euclidean;

fn main() {
    let mut network = neura_network![
        neura_layer!("dense", Relu, 4, 2),
        neura_layer!("dense", Relu, 3),
        neura_layer!("dense", Relu, 1)
    ];

    let inputs = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ];

    for (input, target) in inputs {
        println!("Input: {:?}, target: {}, actual: {:.3}", &input, target[0], network.eval(&input)[0]);
    }

    let mut trainer = NeuraBatchedTrainer::new(0.05, 1000);
    trainer.batch_size = 6;
    trainer.log_epochs = 250;
    trainer.learning_momentum = 0.01;

    trainer.train(
        NeuraBackprop::new(Euclidean),
        &mut network,
        cycle_shuffling(inputs.iter().cloned(), rand::thread_rng()),
        &inputs,
    );

    for (input, target) in inputs {
        println!("Input: {:?}, target: {}, actual: {:.3}", &input, target[0], network.eval(&input)[0]);
    }
}
