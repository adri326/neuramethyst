use std::borrow::Borrow;

use nalgebra::{Const, DVector, Dyn, VecStorage};

use crate::prelude::NeuraShape;

#[derive(Clone, Copy, Debug)]
pub struct NeuraAxisAppend;

#[derive(Clone, Copy, Debug)]
pub enum NeuraAxisErr {
    NoInput,
    ConflictingShape(NeuraShape, NeuraShape),
}

pub trait NeuraCombineInputs<T> {
    type Combined;

    fn combine(&self, inputs: Vec<impl Borrow<T>>) -> Self::Combined;

    // TODO:
    // fn shape(&self, input_shapes: Vec<NeuraShape>) -> NeuraShape;
}

impl<F: Clone> NeuraCombineInputs<DVector<F>> for NeuraAxisAppend {
    type Combined = DVector<F>;

    fn combine(&self, inputs: Vec<impl Borrow<DVector<F>>>) -> Self::Combined {
        assert!(inputs.len() > 0);
        let mut res = Vec::with_capacity(inputs.iter().map(|vec| vec.borrow().len()).sum());

        for input in inputs {
            for x in input.borrow().iter() {
                res.push(x.clone());
            }
        }

        DVector::from_data(VecStorage::new(Dyn(res.len()), Const as Const<1>, res))
    }
}

impl NeuraCombineInputs<NeuraShape> for NeuraAxisAppend {
    type Combined = Result<NeuraShape, NeuraAxisErr>;

    fn combine(&self, inputs: Vec<impl Borrow<NeuraShape>>) -> Self::Combined {
        let mut inputs = inputs.into_iter().map(|x| *x.borrow());
        if let Some(mut res) = inputs.next() {
            for operand in inputs {
                match (res, operand) {
                    (NeuraShape::Vector(x), NeuraShape::Vector(y)) => {
                        res = NeuraShape::Vector(x + y);
                    }
                    (x, y) => {
                        return Err(NeuraAxisErr::ConflictingShape(x, y));
                    }
                }
            }
            Ok(res)
        } else {
            Err(NeuraAxisErr::NoInput)
        }
    }
}
