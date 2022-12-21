use ark_ff::FftField;
use ark_poly::{GeneralEvaluationDomain, EvaluationDomain, Polynomial, univariate::DensePolynomial, UVPolynomial};

use crate::{
    error::Error,
    PolyProcessor
};

pub struct FftProcessor<F: FftField> {
    domain: GeneralEvaluationDomain<F>
}

impl<F: FftField> FftProcessor<F> {
    pub fn construct(domain: GeneralEvaluationDomain<F>) -> Result<Self, Error> {
        if domain.size() & domain.size() - 1 != 0 {
            return Err(Error::NotPow2);
        }
        Ok(Self {
            domain
        })
    }
}

impl<F: FftField> PolyProcessor<F> for FftProcessor<F> {
    fn get_vanishing(&self) -> DensePolynomial<F> {
        self.domain.vanishing_polynomial().into()
    }
    fn evaluate_over_domain(&self, f: &DensePolynomial<F>) -> Vec<F> {
        assert!(f.degree() < self.domain.size());
        self.domain.fft(f)
    }

    fn interpolate(&self, evals: &[F]) -> DensePolynomial<F> {
        assert_eq!(evals.len(), self.domain.size());
        DensePolynomial::from_coefficients_slice(&self.domain.ifft(evals))
    }

    fn batch_evaluate_lagrange_basis(&self, point: &F) -> Vec<F> {
        self.domain.evaluate_all_lagrange_coefficients(*point)
    }
}