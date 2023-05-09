use std::borrow::Borrow;
use std::fmt::Debug;

use nalgebra::{Const, DVector, Dyn, Scalar, VecStorage};

use crate::err::NeuraAxisErr;
use crate::prelude::NeuraShape;

pub trait NeuraAxisBase: Clone + Debug + 'static {
    type Err: Debug;

    fn shape(&self, input_shapes: &[NeuraShape]) -> Result<NeuraShape, Self::Err>;
}

/// Axis operators take in a set of inputs, and combine them together into one output,
/// which is then usually fed to a layer.
pub trait NeuraAxis<Input>: NeuraAxisBase {
    type Combined: 'static;

    fn combine(&self, inputs: &[impl Borrow<Input>]) -> Self::Combined;

    fn split(&self, combined: &Self::Combined, input_shapes: &[NeuraShape]) -> Vec<Input>;
}

/// An axis operator that
#[derive(Clone, Debug)]
pub struct NeuraAxisDefault;

impl NeuraAxisBase for NeuraAxisDefault {
    type Err = NeuraAxisErr;

    fn shape(&self, inputs: &[NeuraShape]) -> Result<NeuraShape, NeuraAxisErr> {
        if inputs.len() != 1 {
            Err(NeuraAxisErr::InvalidAmount(inputs.len(), 1, Some(1)))
        } else {
            Ok(inputs[0])
        }
    }
}

impl<Data: Clone + 'static> NeuraAxis<Data> for NeuraAxisDefault {
    type Combined = Data;

    fn combine(&self, inputs: &[impl Borrow<Data>]) -> Self::Combined {
        assert!(inputs.len() == 1);

        inputs[0].borrow().clone()
    }

    fn split(&self, combined: &Self::Combined, _input_shapes: &[NeuraShape]) -> Vec<Data> {
        vec![combined.clone()]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NeuraAxisAppend;

impl NeuraAxisBase for NeuraAxisAppend {
    type Err = NeuraAxisErr;

    fn shape(&self, inputs: &[NeuraShape]) -> Result<NeuraShape, NeuraAxisErr> {
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

impl<F: Clone + Default + Scalar> NeuraAxis<DVector<F>> for NeuraAxisAppend {
    type Combined = DVector<F>;

    fn combine(&self, inputs: &[impl Borrow<DVector<F>>]) -> Self::Combined {
        assert!(inputs.len() > 0);
        let mut res = Vec::with_capacity(inputs.iter().map(|vec| vec.borrow().len()).sum());

        for input in inputs {
            for x in input.borrow().iter() {
                res.push(x.clone());
            }
        }

        DVector::from_data(VecStorage::new(Dyn(res.len()), Const as Const<1>, res))
    }

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
