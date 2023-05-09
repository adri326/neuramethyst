//! Various error types
//!

use std::fmt::{Debug, Formatter};

use crate::prelude::*;

pub trait NeuraRecursiveErrDebug {
    fn fmt_rec(&self, f: &mut Formatter<'_>, depth: usize) -> std::fmt::Result;
}

impl<Err: Debug> NeuraRecursiveErrDebug for NeuraRecursiveErr<Err, ()> {
    fn fmt_rec(&self, f: &mut Formatter<'_>, depth: usize) -> std::fmt::Result {
        match self {
            Self::Current(err) => {
                write!(f, "NeuraRecursiveErr(depth={}, ", depth)?;
                err.fmt(f)?;
                write!(f, ")")
            }
            Self::Child(_) => write!(f, "NeuraRecursiveErr(depth={}, ())", depth),
        }
    }
}

impl<Err: Debug, ChildErr: NeuraRecursiveErrDebug> NeuraRecursiveErrDebug
    for NeuraRecursiveErr<Err, ChildErr>
{
    #[inline]
    fn fmt_rec(&self, f: &mut Formatter<'_>, depth: usize) -> std::fmt::Result {
        match self {
            Self::Current(err) => {
                write!(f, "NeuraRecursiveErr(depth={}, ", depth)?;
                err.fmt(f)?;
                write!(f, ")")
            }
            Self::Child(err) => err.fmt_rec(f, depth + 1),
        }
    }
}

impl<Err: Debug, ChildErr> Debug for NeuraRecursiveErr<Err, ChildErr>
where
    Self: NeuraRecursiveErrDebug,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt_rec(f, 0)
    }
}

/// Error type returned by `NeuraIsolateLayer::construct`
#[derive(Clone, Debug)]
pub enum NeuraIsolateLayerErr {
    Incompatible {
        start: NeuraShape,
        end: NeuraShape,
        input_shape: NeuraShape,
    },
    OutOfBound {
        start: NeuraShape,
        end: NeuraShape,
        input_shape: NeuraShape,
    },
    OutOfOrder {
        start: NeuraShape,
        end: NeuraShape,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum NeuraAxisErr {
    NoInput,
    ConflictingShape(NeuraShape, NeuraShape),
    InvalidAmount(usize, usize, Option<usize>),
}

#[derive(Clone, Debug)]
pub enum NeuraResidualConstructErr<LayerErr, AxisErr> {
    Layer(LayerErr),
    WrongConnection(isize),
    AxisErr(AxisErr),
    NoOutput,
}

#[derive(Clone)]
pub enum NeuraRecursiveErr<Err, ChildErr> {
    Current(Err),
    Child(ChildErr),
}

pub struct NeuraDimensionsMismatch {
    pub existing: usize,
    pub new: NeuraShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeuraGraphErr {
    MissingNode(String),
    InvalidName(String),
    LayerErr(String),
    Cyclic,
}
