# NeurAmethyst

A neural network library written in [Rust](https://www.rust-lang.org/) and for Rust, that focuses on flexibility and ease of use.

```rust
use neuramethyst::prelude::*;
use neuramethyst::derivable::loss::CrossEntropy;

// Create the network
let network = neura_sequential![
    neura_layer!("dense", 100),
    neura_layer!("dropout", 0.5),
    neura_layer!("dense", 40),
    neura_layer!("dropout", 0.5),
    neura_layer!("dense", 10),
    neura_layer!("softmax"),
];

// Assemble the network together, allowing layers to infer the shape of the input data
let mut network = network.construct(NeuraShape::Vector(100)).unwrap();

// Train the network
let trainer = NeuraBatchedTrainer::new()
    .learning_rate(0.03)
    .batch_size(128)
    .epochs(20, 50000); // number of epochs and size of the training set

trainer.train(
    &NeuraBackprop::new(CrossEntropy),
    &mut network,
    input_data(),
    test_data(),
);
```
