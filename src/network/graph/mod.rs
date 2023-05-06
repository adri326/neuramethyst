#![allow(dead_code)] // TODO: remove this

use std::collections::{HashMap, HashSet, VecDeque};

use crate::prelude::*;
use crate::{err::NeuraGraphErr, layer::NeuraShapedLayer};

mod node;
pub use node::*;

pub struct NeuraGraphPartial<Data> {
    pub nodes: Vec<Box<dyn NeuraGraphNodePartial<Data>>>,
    pub output: String,
    pub input: String,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
enum GraphIndex {
    Input,
    Node(usize),
}

impl<Data> NeuraGraphPartial<Data> {
    fn get_index_map(&self) -> Result<HashMap<String, GraphIndex>, NeuraGraphErr> {
        let mut result = HashMap::with_capacity(self.nodes.len());

        result.insert(self.input.clone(), GraphIndex::Input);

        for (index, node) in self.nodes.iter().enumerate() {
            if result.contains_key(node.name()) {
                return Err(NeuraGraphErr::InvalidName(node.name().to_string()));
            }
            result.insert(node.name().to_string(), GraphIndex::Node(index));
        }

        Ok(result)
    }

    fn get_reverse_graph(
        &self,
        index_map: &HashMap<String, GraphIndex>,
    ) -> Result<HashMap<GraphIndex, HashSet<GraphIndex>>, NeuraGraphErr> {
        let mut result = HashMap::new();

        result.insert(GraphIndex::Input, HashSet::new());

        for i in 0..self.nodes.len() {
            result.insert(GraphIndex::Node(i), HashSet::new());
        }

        for (index, node) in self.nodes.iter().enumerate() {
            for input in node.inputs() {
                let input_index = index_map
                    .get(input)
                    .copied()
                    .ok_or_else(|| NeuraGraphErr::MissingNode(input.clone()))?;
                result
                    .get_mut(&input_index)
                    .expect("index_map returned invalid values")
                    .insert(GraphIndex::Node(index));
            }
        }

        Ok(result)
    }

    fn get_node_order(
        &self,
        index_map: &HashMap<String, GraphIndex>,
        reverse_graph: &HashMap<GraphIndex, HashSet<GraphIndex>>,
    ) -> Result<Vec<usize>, NeuraGraphErr> {
        let mut result: Vec<usize> = Vec::new();
        let mut closed: HashSet<GraphIndex> = HashSet::with_capacity(self.nodes.len());
        let mut open = VecDeque::with_capacity(self.nodes.len());
        open.push_front(GraphIndex::Input);

        /*
        index_map.get(&self.output)
            .copied()
            .ok_or_else(|| NeuraGraphErr::MissingNode(self.output.clone()))?
        */

        while let Some(current) = open.pop_back() {
            if closed.contains(&current) {
                continue;
            }

            closed.insert(current);
            // Do not put 0 (the input) in result
            if let GraphIndex::Node(index) = current {
                result.push(index);
            }

            println!("{:?}", current);

            for next_node in reverse_graph[&current].iter().copied() {
                // Ignore nodes that are already in the closed set
                if closed.contains(&next_node) {
                    continue;
                }

                let GraphIndex::Node(node_index) = next_node else {
                    panic!("Unreachable: cannot have GraphIndex::Input as the output of a node");
                };

                let inputs = self.nodes[node_index].inputs();

                // Only consider nodes whose inputs are in the closed set (meaning they would be ready to be evaluated)
                if !inputs
                    .iter()
                    .all(|input| closed.contains(&index_map[input]))
                {
                    continue;
                }

                open.push_front(next_node);
            }
        }

        if result.len() != self.nodes.len() {
            // TODO: verify that if result.len() != self.nodes.len(), then there is a cyclic subgraph

            return Err(NeuraGraphErr::Cyclic);
        }

        Ok(result)
    }
}

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

    input_shape: NeuraShape,
    output_shape: NeuraShape,

    output_index: usize,
    buffer_size: usize,
}

impl<Data> NeuraShapedLayer for NeuraGraph<Data> {
    fn output_shape(&self) -> NeuraShape {
        self.output_shape
    }
}

impl<Data> NeuraPartialLayer for NeuraGraphPartial<Data> {
    type Constructed = NeuraGraph<Data>;

    type Err = NeuraGraphErr;

    fn construct(self, input_shape: NeuraShape) -> Result<Self::Constructed, Self::Err> {
        let index_map = self.get_index_map()?;
        let reverse_graph = self.get_reverse_graph(&index_map)?;

        // List out the nodes in their execution order
        let node_order = self.get_node_order(&index_map, &reverse_graph)?;
        let mut new_index_map: HashMap<String, usize> = HashMap::from_iter(
            node_order
                .iter()
                .map(|&i| (self.nodes[i].name().to_string(), i)),
        );
        new_index_map.insert(self.input.clone(), 0);

        // TODO: filter out the nodes that are not necessary for computing the result (BFS from the output node back to the inputs)
        // A temporary solution can be to trim the graph
        let output_index = new_index_map
            .get(&self.output)
            .copied()
            .ok_or_else(|| NeuraGraphErr::MissingNode(self.output.clone()))?;

        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut shapes: Vec<Option<NeuraShape>> = vec![None; self.nodes.len() + 1];
        shapes[0] = Some(input_shape);

        for index in node_order.into_iter() {
            let node = &*self.nodes[index];
            let node_inputs = node.inputs();
            let mut inputs = Vec::with_capacity(node_inputs.len());
            let mut input_shapes = Vec::with_capacity(node_inputs.len());

            for input in node_inputs {
                let input_index = new_index_map.get(input).copied().expect(
                    "Unreachable: new_index_map should contain all nodes defined and all nodes should have existing nodes as input"
                );
                inputs.push(input_index);
                input_shapes.push(shapes[input_index].expect(
                    "Unreachable: the order of execution should guarantee that all inputs have appeared before")
                );
            }

            let (constructed, output_shape) = node
                .construct(input_shapes)
                .map_err(|e| NeuraGraphErr::LayerErr(e))?;

            shapes[index] = Some(output_shape);

            nodes.push(NeuraGraphNodeConstructed {
                node: constructed,
                inputs,
                output: new_index_map
                    .get(node.name())
                    .copied()
                    .unwrap_or_else(|| unreachable!()),
            });
        }

        let output_shape = shapes[output_index].unwrap_or_else(|| unreachable!());

        Ok(NeuraGraph {
            nodes,
            input_shape,
            output_shape,
            output_index,
            buffer_size: self.nodes.len() + 1,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::network::residual::NeuraAxisAppend;

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
}
