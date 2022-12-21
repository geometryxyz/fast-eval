use std::{marker::PhantomData, vec};

use ark_ff::FftField;
use ark_poly::{
    univariate::{DenseOrSparsePolynomial, DensePolynomial},
    Polynomial, UVPolynomial,
};

pub struct FastEval<F: FftField> {
    _f: PhantomData<F>,
}

impl<F: FftField> FastEval<F> {
    pub fn divide_down_the_tree(
        layers: &Vec<Vec<DensePolynomial<F>>>,
        n: usize,
        root: (usize, usize),
        f: &DensePolynomial<F>,
    ) -> Vec<F> {
        assert!(f.degree() < n);

        if n == 1 {
            return vec![f.coeffs[0]];
        }

        let f_ds = DenseOrSparsePolynomial::from(f);
        let lhs_divisor = DenseOrSparsePolynomial::from(&layers[root.0 - 1][2 * root.1]);
        let rhs_divisor = DenseOrSparsePolynomial::from(&layers[root.0 - 1][2 * root.1 + 1]);

        let (_, r0) = f_ds.divide_with_q_and_r(&lhs_divisor).unwrap();
        let (_, r1) = f_ds.divide_with_q_and_r(&rhs_divisor).unwrap();

        let mut lhs_evals =
            Self::divide_down_the_tree(layers, n / 2, (root.0 - 1, 2 * root.1), &r0);
        let rhs_evals =
            Self::divide_down_the_tree(layers, n / 2, (root.0 - 1, 2 * root.1 + 1), &r1);

        lhs_evals.extend_from_slice(&rhs_evals);
        lhs_evals
    }

    pub fn multiply_up_the_tree(
        layers: &Vec<Vec<DensePolynomial<F>>>,
        index_bounds: (usize, usize),
        root: (usize, usize),
        evals: &Vec<F>,
    ) -> DensePolynomial<F> {
        if index_bounds.1 - index_bounds.0 == 0 {
            return DensePolynomial::from_coefficients_slice(&[evals[index_bounds.0]]);
        }

        let len = (index_bounds.1 - index_bounds.0) / 2;
        let lhs_bounds = (index_bounds.0, index_bounds.0 + len);
        let rhs_bounds = (lhs_bounds.1 + 1, index_bounds.1);

        let r0 = Self::multiply_up_the_tree(layers, lhs_bounds, (root.0 - 1, 2 * root.1), evals);
        let r1 =
            Self::multiply_up_the_tree(layers, rhs_bounds, (root.0 - 1, 2 * root.1 + 1), evals);

        let lhs = &layers[root.0 - 1][2 * root.1];
        let rhs = &layers[root.0 - 1][2 * root.1 + 1];
        &r0 * rhs + &r1 * lhs
    }
}
