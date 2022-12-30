use std::{marker::PhantomData, vec};

use ark_ff::{FftField, Zero};
use ark_poly::{univariate::DensePolynomial, Polynomial, UVPolynomial};

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

        //let f_ds = DenseOrSparsePolynomial::from(f);
        let lhs_divisor = layers[root.0 - 1][2 * root.1].clone();
        let rhs_divisor = layers[root.0 - 1][2 * root.1 + 1].clone();

        let (_, r0) = Self::fast_divide_with_q_and_r(f, &lhs_divisor).unwrap();
        let (_, r1) = Self::fast_divide_with_q_and_r(f, &rhs_divisor).unwrap();

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

    #[allow(non_snake_case)]
    //for [f(X),l] outputs [g(X)] such that f(X)* g(X)= 1 mod X^l
    fn poly_inverse(poly: &DensePolynomial<F>, l: u32) -> Option<DensePolynomial<F>> {
        if poly.is_zero() {
            panic!("Dividing by zero polynomial")
        } else {
            let mut g = DensePolynomial::from_coefficients_slice(&[F::one() / poly.coeffs[0]]); //g0=f(0)
            let mut i = 1;
            let two = F::one() + F::one();
            while (1 << i) < 2 * l {
                //i ranges from 1 to ceiling (log_2 l)
                let tmp = (&g * two) + &(poly * &(&g * &g)) * (-F::one()); //g_{i+1} = (2g_i - f g_i^2)\bmod{x^{2^{i+1}}}
                let mut a = tmp.coeffs().to_vec();
                a.resize(1 << i, F::zero()); //mod x^(2^i)
                g = DensePolynomial::from_coefficients_vec(a);
                i = i + 1;
            }
            Some(g)
        }
    }

    fn poly_reverse(poly: &DensePolynomial<F>) -> DensePolynomial<F> {
        let vec_coeff = poly.coeffs().to_vec();

        let mut x = vec![];
        x.resize(poly.degree() + 1, F::one());
        for i in 0..poly.degree() + 1 {
            x[i] = vec_coeff[poly.degree() - i];
        }
        let out = DensePolynomial::from_coefficients_vec(x);
        out
    }

    //for p(X)  outputs p(X) mod X^l
    fn poly_trim(poly: &DensePolynomial<F>, l: usize) -> DensePolynomial<F> {
        let mut vec_coeff = poly.coeffs().to_vec();
        vec_coeff.resize(l, F::zero());
        return DensePolynomial::from_coefficients_vec(vec_coeff);
    }

    #[allow(non_snake_case)]
    //for [p(X), g(X)] outputs [q(X),r(X)] such that p(X) = g(X)q(X)+r(X)
    pub fn fast_divide_with_q_and_r(
        poly: &DensePolynomial<F>,
        divisor: &DensePolynomial<F>,
    ) -> Option<(DensePolynomial<F>, DensePolynomial<F>)> {
        if poly.is_zero() {
            Some((DensePolynomial::zero(), DensePolynomial::zero()))
        } else if divisor.is_zero() {
            panic!("Dividing by zero polynomial")
        } else if poly.degree() < divisor.degree() {
            Some((DensePolynomial::zero(), poly.clone().into()))
        } else {
            // Now we know that self.degree() >= divisor.degree();
            // Use the formula for q: rev(q)=  rev(f)* rev(g)^{-1} mod x^{deg(f)-deg(g)+1}.
            let rev_f = Self::poly_reverse(poly); //reverse of f
            let rev_g = Self::poly_reverse(divisor); //reverse of g
            let inv_rev_g = Self::poly_inverse(&rev_g, poly.degree() as u32 + 1).unwrap();
            let tmp = &rev_f * &inv_rev_g;
            let rev_q = Self::poly_trim(&tmp, poly.degree() - divisor.degree() + 1);
            let quotient = Self::poly_reverse(&rev_q);
            let remainder = poly + &(&(&quotient * divisor) * (-F::one()));

            Some((quotient, remainder))
        }
    }
}

//////////////////////////////////////////////////////

#[cfg(test)]
pub mod tests {

    use crate::fast_eval::FastEval;

    use ark_poly::{
        univariate::DenseOrSparsePolynomial, univariate::DensePolynomial, Polynomial, UVPolynomial,
    };

    use ark_bn254::Fr;
    use ark_ff::One;
    use std::time::Instant;

    #[allow(non_snake_case)]
    #[test]
    pub fn test_poly_inverse() {
        let max_m: u32 = 20;
        let rng = &mut ark_std::test_rng();
        let poly_one = DensePolynomial::from_coefficients_vec(vec![Fr::one()]);

        for i in 1..max_m {
            println!("inverse of degree {:?}", i);
            let c_poly = DensePolynomial::<Fr>::rand(i as usize, rng);
            for l in i + 1..max_m {
                let inv = FastEval::poly_inverse(&c_poly, l).unwrap();
                let prod = &c_poly * &inv;
                let res = FastEval::poly_trim(&prod, l as usize);
                assert_eq!(poly_one, res);
            }
        }
    }

    #[allow(non_snake_case)]
    #[test]
    pub fn test_poly_reverse() {
        let max_m: u32 = 20;
        let rng = &mut ark_std::test_rng();
        let c_poly = DensePolynomial::<Fr>::rand(max_m as usize, rng);
        let rev = FastEval::poly_reverse(&c_poly);
        let res = FastEval::poly_reverse(&rev);
        assert_eq!(res, c_poly)
    }

    #[allow(non_snake_case)]
    #[test]
    pub fn test_fast_poly_division() {
        let max_m: u32 = 20;
        let rng = &mut ark_std::test_rng();

        for i in 2..max_m {
            println!("division of degree {:?}", i);
            let c_poly = DensePolynomial::<Fr>::rand(i as usize, rng);
            let g_poly = DensePolynomial::<Fr>::rand((i >> 1) as usize, rng);
            let (q, r) = FastEval::fast_divide_with_q_and_r(&c_poly, &g_poly).unwrap();
            let res = (&q * &g_poly) + r.clone();
            assert_eq!(c_poly, res);
            assert!(r.degree() < g_poly.degree());
        }
    }

    #[allow(non_snake_case)]
    #[test]
    pub fn compare_fast_poly_division() {
        let max_m: u32 = 10;
        let rng = &mut ark_std::test_rng();

        for j in 1..max_m {
            let i = 1 << j;
            println!("division of degree {:?}", i);
            let c_poly = DensePolynomial::<Fr>::rand(i as usize, rng);
            let c_poly_sp = DenseOrSparsePolynomial::from(c_poly.clone());
            let g_poly = DensePolynomial::<Fr>::rand((i >> 1) as usize, rng);
            let g_poly_sp = DenseOrSparsePolynomial::from(g_poly.clone());
            let now = Instant::now();
            let (q, r) = FastEval::fast_divide_with_q_and_r(&c_poly, &g_poly).unwrap();
            println!("Time to fast divide   {:?}", now.elapsed());
            let _now2 = Instant::now();
            let (_q2, _r2) = c_poly_sp.divide_with_q_and_r(&g_poly_sp).unwrap();
            println!("Time to regular divide   {:?}", now.elapsed());
            let res = (&q * &g_poly) + r;
            assert_eq!(c_poly, res);
        }
    }

    #[allow(non_snake_case)]
    #[test]
    pub fn test_evaluations() {
        let max_m: u32 = 10;

        let rng = &mut ark_std::test_rng();

        for j in 1..max_m {
            let i = 1 << j;
            println!("division of degree {:?}", i);
            let _c_poly = DensePolynomial::<Fr>::rand(i as usize, rng);
            //let c_poly_sp = DenseOrSparsePolynomial::from( c_poly.clone() );
        }
    }
}
