use std::any::Any;

use crate::{
    algebra::NeuraDynVectorSpace, derivable::NeuraLoss, layer::NeuraLayerBase, prelude::*,
};

mod node;
pub use node::*;

mod partial;
pub use partial::NeuraGraphPartial;

mod from;
pub use from::FromSequential;

mod backprop;
pub use backprop::NeuraGraphBackprop;

#[derive(Debug)]
pub struct NeuraGraphNodeConstructed<Data> {
    node: Box<dyn NeuraGraphNodeEval<Data>>,
    inputs: Vec<usize>,
    output: usize,
}

impl<Data> Clone for NeuraGraphNodeConstructed<Data> {
    fn clone(&self) -> Self {
        Self {
            node: dyn_clone::clone_box(&*self.node),
            inputs: self.inputs.clone(),
            output: self.output,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NeuraGraph<Data> {
    /// ## Class invariants
    ///
    /// - The order of nodes should match with the order of execution, ie.
    ///   `forall (x, y), nodes = [..., x, ..., y, ...] => !(y in x.node.inputs)`
    /// - `nodes[0].inputs = [0]`
    nodes: Vec<NeuraGraphNodeConstructed<Data>>,

    // input_shape: NeuraShape,
    output_shape: NeuraShape,

    output_index: usize,
    buffer_size: usize,
}

impl<Data: Clone + std::fmt::Debug + 'static> NeuraLayerBase for NeuraGraph<Data> {
    type Gradient = Vec<Box<dyn NeuraDynVectorSpace>>;

    fn output_shape(&self) -> NeuraShape {
        self.output_shape
    }

    fn default_gradient(&self) -> Self::Gradient {
        let mut res: Self::Gradient = Vec::with_capacity(self.buffer_size);

        res.push(Box::new(()));

        for node in self.nodes.iter() {
            res.push(node.node.default_gradient());
        }

        res
    }

    fn apply_gradient(&mut self, gradient: &Self::Gradient) {
        for (node, gradient) in self.nodes.iter_mut().zip(gradient.iter()) {
            node.node.apply_gradient(gradient);
        }
    }

    fn prepare_layer(&mut self, is_training: bool) {
        for node in self.nodes.iter_mut() {
            node.node.prepare(is_training);
        }
    }

    fn regularize_layer(&self) -> Self::Gradient {
        let mut res: Self::Gradient = Vec::with_capacity(self.buffer_size);

        res.push(Box::new(()));

        for node in self.nodes.iter() {
            res.push(node.node.get_regularization_gradient());
        }

        res
    }
}

impl<Data> NeuraGraph<Data> {
    fn create_buffer<T>(&self) -> Vec<Option<T>> {
        let mut res = Vec::with_capacity(self.buffer_size);

        for _ in 0..self.buffer_size {
            res.push(None);
        }

        res
    }

    fn eval_in(&self, input: &Data, buffer: &mut [Option<Data>])
    where
        Data: Clone,
    {
        assert!(buffer.len() >= self.nodes.len());

        buffer[0] = Some(input.clone());

        for node in self.nodes.iter() {
            // PERF: re-use the allocation for `inputs`, and `.take()` the elements only needed once?
            let inputs: Vec<_> = node
                .inputs
                .iter()
                .map(|&i| {
                    buffer[i]
                        .clone()
                        .expect("Unreachable: output of previous layer was not set")
                })
                .collect();
            let result = node.node.eval(&inputs);
            buffer[node.output] = Some(result);
        }
    }
}

impl<Data: Clone + std::fmt::Debug + 'static> NeuraLayer<Data> for NeuraGraph<Data> {
    type Output = Data;
    type IntermediaryRepr = ();

    fn eval(&self, input: &Data) -> Self::Output {
        let mut buffer = self.create_buffer();

        self.eval_in(input, &mut buffer);

        buffer[self.output_index]
            .take()
            .expect("Unreachable: output was not set")
    }

    #[allow(unused)]
    fn eval_training(&self, input: &Data) -> (Self::Output, Self::IntermediaryRepr) {
        unimplemented!("NeuraGraph cannot be used as a trainable layer yet");
    }

    #[allow(unused)]
    fn backprop_layer(
        &self,
        input: &Data,
        intermediary: &Self::IntermediaryRepr,
        epsilon: &Self::Output,
    ) -> Data {
        unimplemented!("NeuraGraph cannot be used as a trainable layer yet");
    }
}

#[cfg(test)]
mod test {
    use crate::{axis::NeuraAxisAppend, err::NeuraGraphErr, utils::uniform_vector};

    use super::*;

    #[test]
    fn test_construct_simple_graph() {
        let graph = NeuraGraphPartial {
            nodes: vec![NeuraGraphNode::new(
                vec!["input".to_string()],
                NeuraAxisAppend,
                neura_layer!("dense", 10),
                "output".to_string(),
            )
            .as_boxed()],
            output: "output".to_string(),
            input: "input".to_string(),
        };

        let constructed = graph.construct(NeuraShape::Vector(5));

        assert!(constructed.is_ok());
    }

    #[test]
    fn test_construct_deep_graph() {
        let graph = NeuraGraphPartial {
            nodes: vec![
                // Node intentionally out of order
                NeuraGraphNode::new(
                    vec!["inter".to_string(), "inter2".to_string()],
                    NeuraAxisAppend,
                    neura_layer!("dense", 2),
                    "output".to_string(),
                )
                .as_boxed(),
                NeuraGraphNode::new(
                    vec!["input".to_string()],
                    NeuraAxisAppend,
                    neura_layer!("dense", 10),
                    "inter".to_string(),
                )
                .as_boxed(),
                NeuraGraphNode::new(
                    vec!["inter".to_string()],
                    NeuraAxisAppend,
                    neura_layer!("dense", 20),
                    "inter2".to_string(),
                )
                .as_boxed(),
            ],
            output: "output".to_string(),
            input: "input".to_string(),
        };

        let index_map = graph.get_index_map().unwrap();
        let reverse_graph = graph.get_reverse_graph(&index_map).unwrap();
        assert_eq!(
            graph.get_node_order(&index_map, &reverse_graph),
            Ok(vec![1, 2, 0])
        );

        let constructed = graph.construct(NeuraShape::Vector(5));

        assert!(constructed.is_ok());
    }

    #[test]
    fn test_construct_cyclic_graph() {
        let graph = NeuraGraphPartial {
            nodes: vec![NeuraGraphNode::new(
                vec!["input".to_string(), "output".to_string()],
                NeuraAxisAppend,
                neura_layer!("dense", 10),
                "output".to_string(),
            )
            .as_boxed()],
            output: "output".to_string(),
            input: "input".to_string(),
        };

        let constructed = graph.construct(NeuraShape::Vector(5));

        assert_eq!(constructed.unwrap_err(), NeuraGraphErr::Cyclic);
    }

    #[test]
    fn test_construct_disjoint_graph() {
        let graph = NeuraGraphPartial {
            nodes: vec![
                NeuraGraphNode::new(
                    vec!["input".to_string()],
                    NeuraAxisAppend,
                    neura_layer!("dense", 10),
                    "inter".to_string(),
                )
                .as_boxed(),
                NeuraGraphNode::new(
                    vec!["missing".to_string()],
                    NeuraAxisAppend,
                    neura_layer!("dense", 10),
                    "output".to_string(),
                )
                .as_boxed(),
            ],
            output: "output".to_string(),
            input: "input".to_string(),
        };

        let constructed = graph.construct(NeuraShape::Vector(5));

        assert_eq!(
            constructed.unwrap_err(),
            NeuraGraphErr::MissingNode(String::from("missing"))
        );
    }

    #[test]
    fn test_eval_equal_sequential() {
        let network = neura_sequential![
            neura_layer!("dense", 4, f64),
            neura_layer!("dense", 2, f64),
            neura_layer!("softmax")
        ]
        .construct(NeuraShape::Vector(3))
        .unwrap();

        let graph = NeuraGraph::from_sequential(network.clone(), NeuraShape::Vector(3));

        for _ in 0..10 {
            let input = uniform_vector(3);
            let seq_result = network.eval(&input);
            let graph_result = graph.eval(&input);

            assert_eq!(seq_result.shape(), graph_result.shape());
            approx::assert_relative_eq!(seq_result[0], graph_result[0]);
            approx::assert_relative_eq!(seq_result[1], graph_result[1]);
        }
    }
}
