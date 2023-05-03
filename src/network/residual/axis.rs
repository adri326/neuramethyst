use std::borrow::Borrow;

use nalgebra::{Const, DVector, Dyn, Scalar, VecStorage};

use crate::{err::NeuraAxisErr, prelude::NeuraShape};

#[derive(Clone, Copy, Debug)]
pub struct NeuraAxisAppend;

pub trait NeuraCombineInputs<T> {
    type Combined;

    fn combine(&self, inputs: Vec<impl Borrow<T>>) -> Self::Combined;
}

pub trait NeuraSplitInputs<T>: NeuraCombineInputs<T> {
    fn split(&self, combined: &Self::Combined, input_shapes: &[NeuraShape]) -> Vec<T>;
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

impl<F: Clone + Scalar + Default> NeuraSplitInputs<DVector<F>> for NeuraAxisAppend {
    fn split(&self, combined: &Self::Combined, input_shapes: &[NeuraShape]) -> Vec<DVector<F>> {
        let mut result = Vec::with_capacity(input_shapes.len());
        let mut offset = 0;

        for &input_shape in input_shapes.iter() {
            let NeuraShape::Vector(input_shape) = input_shape else {
                panic!("Expected {:?} to be NeuraShape::Vector", input_shape);
            };

            let mut subvector = DVector::from_element(input_shape, F::default());

            for i in 0..input_shape {
                subvector[i] = combined[offset + i].clone();
            }
            result.push(subvector);

            offset += input_shape;
        }

        result
    }
}
