use crate::{layer::NeuraShapedLayer, prelude::*};

mod node;
pub use node::*;

mod partial;
pub use partial::NeuraGraphPartial;

mod from;

#[derive(Debug)]
struct NeuraGraphNodeConstructed<Data> {
    node: Box<dyn NeuraGraphNodeEval<Data>>,
    inputs: Vec<usize>,
    output: usize,
}

#[derive(Debug)]
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

impl<Data> NeuraShapedLayer for NeuraGraph<Data> {
    fn output_shape(&self) -> NeuraShape {
        self.output_shape
    }
}

impl<Data> NeuraGraph<Data> {
    fn create_buffer(&self) -> Vec<Option<Data>> {
        let mut res = Vec::with_capacity(self.buffer_size);

        for _ in 0..self.buffer_size {
            res.push(None);
        }

        res
    }

    fn eval_in(&self, input: &Data, buffer: &mut Vec<Option<Data>>)
    where
        Data: Clone,
    {
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

impl<Data: Clone> NeuraLayer<Data> for NeuraGraph<Data> {
    type Output = Data;

    fn eval(&self, input: &Data) -> Self::Output {
        let mut buffer = self.create_buffer();

        self.eval_in(input, &mut buffer);

        buffer[self.output_index]
            .take()
            .expect("Unreachable: output was not set")
    }
}

#[cfg(test)]
mod test {
    use crate::{err::NeuraGraphErr, network::residual::NeuraAxisAppend, utils::uniform_vector};

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

        let graph = NeuraGraph::from(network.clone());

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
