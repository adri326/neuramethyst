#![feature(generic_arg_infer)]

use neuramethyst::prelude::*;
use neuramethyst::derivable::activation::{Relu, Tanh};
use neuramethyst::derivable::loss::Euclidean;

fn main() {
    let mut network = neura_network![
        neura_layer!("dense", Tanh, 2, 2),
        neura_layer!("dense", Tanh, 3),
        neura_layer!("dense", Relu, 1)
    ];

    let inputs = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ];

    // println!("{:#?}", network);

    for (input, target) in inputs {
        println!("Input: {:?}, target: {}, actual: {}", &input, target[0], network.eval(&input)[0]);
    }

    train_batched(
        &mut network,
        inputs.clone(),
        &inputs,
        NeuraBackprop::new(Euclidean),
        0.01,
        1,
        25
    );

    // println!("{:#?}", network);

    for (input, target) in inputs {
        println!("Input: {:?}, target: {}, actual: {}", &input, target[0], network.eval(&input)[0]);
    }
}
