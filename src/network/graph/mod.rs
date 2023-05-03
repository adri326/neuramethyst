use std::collections::{HashMap, HashSet, VecDeque};

use crate::prelude::*;
use crate::{err::NeuraGraphErr, layer::NeuraShapedLayer};

mod node;
pub use node::*;

pub struct NeuraGraphPartial<Data> {
    pub nodes: Vec<Box<dyn NeuraGraphNodeTrait<Data>>>,
    pub output: String,
    pub input: String,
}

impl<Data> NeuraGraphPartial<Data> {
    fn get_index_map(&self) -> Result<HashMap<String, usize>, NeuraGraphErr> {
        let mut result = HashMap::with_capacity(self.nodes.len());

        result.insert(self.input.clone(), 0);

        for (index, node) in self.nodes.iter().enumerate() {
            if result.contains_key(node.name()) {
                return Err(NeuraGraphErr::InvalidName(node.name().to_string()));
            }
            result.insert(node.name().to_string(), index + 1);
        }

        Ok(result)
    }

    fn get_reverse_graph(
        &self,
        index_map: &HashMap<String, usize>,
    ) -> Result<Vec<HashSet<usize>>, NeuraGraphErr> {
        let mut result = vec![HashSet::new(); self.nodes.len()];

        for (index, node) in self.nodes.iter().enumerate() {
            for input in node.inputs() {
                let input_index = index_map
                    .get(input)
                    .copied()
                    .ok_or_else(|| NeuraGraphErr::MissingNode(input.clone()))?;
                result[input_index].insert(index + 1);
            }
        }

        Ok(result)
    }

    fn get_node_order(
        &self,
        index_map: &HashMap<String, usize>,
        reverse_graph: &Vec<HashSet<usize>>,
    ) -> Result<Vec<usize>, NeuraGraphErr> {
        let mut result: Vec<usize> = Vec::new();
        let mut closed: HashSet<usize> = HashSet::with_capacity(self.nodes.len());
        let mut open = VecDeque::with_capacity(self.nodes.len());
        open.push_front(0usize);

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
            result.push(current);

            for output_index in reverse_graph[current].iter().copied() {
                assert!(output_index > 0);

                // Ignore nodes that are already in the closed set
                if closed.contains(&output_index) {
                    continue;
                }

                let inputs = self.nodes[output_index - 1].inputs();

                // Only consider nodes whose inputs are in the closed set
                if !inputs
                    .iter()
                    .all(|input| closed.contains(&index_map[input]))
                {
                    continue;
                }
            }
        }

        Ok(result)
    }
}

struct NeuraGraphNodeConstructed<Data> {
    node: Box<dyn NeuraGraphNodeTrait<Data>>,
    inputs: Vec<usize>,
    output: usize,
}

pub struct NeuraGraph<Data> {
    /// ## Class invariants
    ///
    /// - The order of nodes should match with the order of execution, ie.
    ///   `forall (x, y), nodes = [..., x, ..., y, ...] => !(y in x.node.inputs)`
    /// - `nodes[0].inputs = [0]`
    /// - `nodes[nodes.len() - 1].output = buffer.len() - 1`
    nodes: Vec<NeuraGraphNodeConstructed<Data>>,

    input_shape: NeuraShape,
    output_shape: NeuraShape,
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
        todo!()
    }
}
