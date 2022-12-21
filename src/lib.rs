use std::marker::PhantomData;

use ark_ff::FftField;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial,
    UVPolynomial,
};
use error::Error;
use fft::FftProcessor;

pub use crate::subtree::Pow2ProductSubtree;

pub mod error;
pub mod fast_eval;
pub mod subtree;
pub mod fft;

/// Saves one degree of 2 for FFT when a, b are monic polynomials in leading coefficient
/// panics if a or b are not monic and degree 2
pub fn multiply_pow2_monic_polys<F: FftField>(
    a: &DensePolynomial<F>,
    b: &DensePolynomial<F>,
) -> DensePolynomial<F> {
    let deg_a = a.degree();
    let deg_b = b.degree();

    if deg_a != deg_b {
        panic!("deg_a != deg_b, {}, {}", deg_a, deg_b);
    }

    let monic_deg = deg_a;

    if monic_deg & monic_deg - 1 != 0 {
        panic!("Poly a is not degree of 2");
    }

    // it's safe to unwrap since for degree 0 previous check would panic for overflow
    if *a.coeffs.last().unwrap() != F::one() {
        panic!("Poly a is not monic");
    }

    if *b.coeffs.last().unwrap() != F::one() {
        panic!("Poly b is not monic");
    }

    // it's safe to unwrap since monic_deg is pow2
    let domain = GeneralEvaluationDomain::<F>::new(2 * monic_deg).unwrap();

    let a_evals = domain.fft(a);
    let b_evals = domain.fft(b);

    let product_evals: Vec<F> = a_evals
        .iter()
        .zip(b_evals.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    /*
        We know that coefficient of x^(2^m) will be 1 so it will end up in front of x^0,
        That's why we just subtract 1 from free coefficient of resulting poly
    */
    let mut product_poly = DensePolynomial::from_coefficients_slice(&domain.ifft(&product_evals));
    product_poly[0] -= F::one();
    product_poly.coeffs.push(F::one());

    product_poly
}

pub trait PolyProcessor<F: FftField> {
    fn evaluate_over_domain(&self, f: &DensePolynomial<F>) -> Vec<F>;

    fn interpolate(&self, evals: &[F]) -> DensePolynomial<F>;

    fn batch_evaluate_lagrange_basis(&self, point: &F) -> Vec<F>;
}

// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 = 16
// 0       1       2          3        4 = 4
pub struct PolyProcessorStrategy<F: FftField> {
    _f: PhantomData<F>
}

impl<F: FftField> PolyProcessorStrategy<F> {
    pub fn resolve(roots: &[F]) -> Result<Box<dyn PolyProcessor<F>>, Error> {
        let n = roots.len(); 
        let domain = GeneralEvaluationDomain::<F>::new(n).unwrap();

        let omegas: Vec<_> = domain.elements().collect();
        if roots == omegas {
            let fft_processor = FftProcessor::<F>::construct(n)?;
            return Ok(Box::new(fft_processor))
        } else {
            let subtree = Pow2ProductSubtree::construct(roots)?;
            return Ok(Box::new(subtree));
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::One;
    use ark_poly::{univariate::DensePolynomial, UVPolynomial};
    use ark_std::test_rng;

    use super::multiply_pow2_monic_polys;

    #[test]
    fn test_monic_fft() {
        let n = 32;
        let mut rng = test_rng();

        let mut a = DensePolynomial::<Fr>::rand(n, &mut rng);
        a.coeffs[n] = Fr::one();

        let mut b = DensePolynomial::<Fr>::rand(n, &mut rng);
        b.coeffs[n] = Fr::one();

        let product_slow = &a * &b;
        let product_fast = multiply_pow2_monic_polys(&a, &b);
        assert_eq!(product_fast, product_slow);
    }
}
