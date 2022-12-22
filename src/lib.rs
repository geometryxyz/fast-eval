use std::marker::PhantomData;

use ark_ff::FftField;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain,
};
use error::Error;
use fft::FftProcessor;

pub use crate::subtree::Pow2ProductSubtree;

pub mod error;
pub mod fast_eval;
pub mod subtree;
pub mod fft;

pub trait PolyProcessor<F: FftField> {
    fn get_vanishing(&self) -> DensePolynomial<F>;

    fn get_ri(&self) -> Vec<F>;

    fn evaluate_over_domain(&self, f: &DensePolynomial<F>) -> Vec<F>;

    fn interpolate(&self, evals: &[F]) -> DensePolynomial<F>;

    fn batch_evaluate_lagrange_basis(&self, point: &F) -> Vec<F>;
}

pub struct PolyProcessorStrategy<F: FftField> {
    _f: PhantomData<F>
}

impl<F: FftField> PolyProcessorStrategy<F> {
    pub fn resolve(roots: &[F]) -> Result<Box<dyn PolyProcessor<F>>, Error> {
        let n = roots.len(); 
        let domain = GeneralEvaluationDomain::<F>::new(n).unwrap();

        let omegas: Vec<_> = domain.elements().collect();
        if roots == omegas {
            let fft_processor = FftProcessor::<F>::construct(domain)?;
            return Ok(Box::new(fft_processor))
        } else {
            let subtree = Pow2ProductSubtree::construct(roots)?;
            return Ok(Box::new(subtree));
        }
    }
}