#![allow(unused)]

use std::io::{stdout, sink};
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::ops::{Add, Mul, Sub, SubAssign};

use rand::Rng;
use rand::distributions::Uniform;
use rand::rngs::ThreadRng;

use crate::Errors::MatToVecError;

fn gen_random_vec(size: usize, min: f64, max: f64) -> Vec<f64> {
    let mut rng = ThreadRng::default();
    let distrib = Uniform::new(min, max);
    (0..size).map(|_| rng.sample(distrib)).collect()
}

#[derive(Debug)]
pub enum Errors
{
    MatToVecError,
    DimensionMismatchError
}

// Convention used is (COLS, ROWS) for cache perf
#[derive(Clone, Debug)]
pub struct FennMatrix {
    data: Vec<f64>,
    // Stored in row major order
    dim: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct FennVector {
    matrix: FennMatrix
}

impl FennMatrix {
    //Dim: (rows, cols) -> (cols, rows)
    pub fn new(dims: (usize, usize)) -> Self {
        FennMatrix {
            data: vec![0f64; dims.0 * dims.1],
            dim: (dims.1, dims.0),
        }
    }

    pub fn ones(dims: (usize, usize)) -> Self {
        FennMatrix {
            data: vec![1f64; dims.0 * dims.1],
            dim: (dims.1, dims.0),
        }
    }

    pub fn rand(dims: (usize, usize), min: f64, max: f64) -> Self {
        FennMatrix {
            data: gen_random_vec(dims.1*dims.0, min, max),
            dim: (dims.1, dims.0),
        }
    }

    pub fn from_slice<const COLS: usize, const ROWS: usize> (data: &[[f64; COLS]; ROWS]) -> Self {
        FennMatrix {
            data: data.iter()
                      .flatten()
                      .copied()
                      .collect::<Vec<f64>>(),
            dim: (COLS, ROWS),
        }
    }

    pub fn from_vec(data: Vec<f64>, dims: (usize, usize)) -> Self {
        FennMatrix {
            data,
            dim: (dims.1, dims.0)
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        self.dim
    }

    pub fn vec_mult(&self, input: &FennVector) -> FennVector {
        debug_assert_eq!(input.len(),
            self.dim.0,
            "Input vector size ({}x1) is not compatible with layer size ({}x{})",
            input.len(), self.dim.0, self.dim.1);
        let mut out = FennVector::from_vec(vec![0f64; self.dim.0]);
        for i in 0..self.dim.0
        {
            for j in 0..input.len()
            {
                out[i] += input[j]*self[i][j];
            }
        }
        out
    }

    pub fn col_mut<'matrix>(&'matrix mut self, col_index: usize) -> Vec<& 'matrix mut f64>
    {
        assert!(col_index < self.dim.0);
        let col_size = self.dim.1;
        let mut v = Vec::with_capacity(col_size);
        for r in 0..col_size
        {
           let ptr = &mut self[r][col_index] as *mut f64;

           //SAFETY: This is okay as the backing memory is the same as the not mut case
           v.push(unsafe { &mut *ptr });
        }
        v
    }
    pub fn col<'matrix>(&'matrix self, col_index: usize) -> Vec<& 'matrix f64>
    {
        assert!(col_index < self.dim.0);
        let col_size = self.dim.1;
        let mut v = Vec::with_capacity(col_size);
        for r in 0..col_size
        {
           v.push(&self[r][col_index]);
        }
        v
    }

    pub fn print(&self, tabbing: usize) -> std::io::Result<()>
    {
        self.print_internal_io(&mut stdout(), tabbing)
    }

    pub fn transpose(&self) -> FennMatrix
    {
        let mut out = self.clone();
        out.transpose_mut();
        out
    }
    pub fn transpose_mut(&mut self)
    {
        let (cols, rows) = self.dim;
        let mut transposed_data = Vec::with_capacity(self.data.len());

        for c in 0..cols {
            for r in 0..rows {
                transposed_data.push(self.data[r * cols + c]);
            }
        }

        self.data = transposed_data;
        self.dim = (rows, cols);
    }

    pub fn move_to_vec(self) -> FennVector
    {
        assert!(self.dim.0 == 1);
        FennVector::from_vec(self.data)
    }

    pub fn to_vec(&self) -> Result<FennVector, Errors>
    {
        match self.dim.0 == 1
        {
            true => Ok(FennVector::from_vec(self.data.clone())),
            false => Err(MatToVecError)
        }
    }

    pub fn map<F: Fn(&f64) -> f64>(&self, f: F) -> FennMatrix
    {
        FennMatrix::from_vec(self.data.iter().map(f).collect(), (self.dim.1, self.dim.0))
    }

    pub fn mut_map<F: Fn(&mut f64) -> f64>(&mut self, f: F)
    {
        self.data.iter_mut().map(f);
    }

    // In-place: self -= scale * (u outer-product v^T).
    // u is treated as a column vector of length n_rows, v as a column vector of length n_cols.
    pub fn sub_outer_product_scaled(&mut self, scale: f64, u: &FennVector, v: &FennVector) {
        let n_rows = self.dim.1;
        let n_cols = self.dim.0;
        debug_assert_eq!(u.len(), n_rows);
        debug_assert_eq!(v.len(), n_cols);
        let u_data = u.as_slice();
        let v_data = v.as_slice();
        for r in 0..n_rows {
            let scaled_u = scale * u_data[r];
            let row_start = r * n_cols;
            for c in 0..n_cols {
                self.data[row_start + c] -= scaled_u * v_data[c];
            }
        }
    }

    // Returns self^T * v without materializing the transpose.
    pub fn mul_transposed_lhs_vec(&self, v: &FennVector) -> FennVector {
        let n_rows = self.dim.1;
        let n_cols = self.dim.0;
        debug_assert_eq!(v.len(), n_rows);
        let v_data = v.as_slice();
        let mut out = vec![0f64; n_cols];
        for k in 0..n_rows {
            let vk = v_data[k];
            let row_start = k * n_cols;
            for i in 0..n_cols {
                out[i] += self.data[row_start + i] * vk;
            }
        }
        FennVector::from_vec(out)
    }

    // Fused Adam update on `self` with gradient = u outer-product v_vec^T.
    // Updates m and v_buf (EMAs of grad and grad^2) in place, then applies
    //   self -= lr * (m / bias_corr1) / (sqrt(v_buf / bias_corr2) + eps)
    // bias_corr1 = 1 - beta1^step, bias_corr2 = 1 - beta2^step (caller-provided).
    pub fn adam_update_outer(
        &mut self,
        m: &mut FennMatrix,
        v_buf: &mut FennMatrix,
        u: &FennVector,
        v_vec: &FennVector,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) {
        let n_rows = self.dim.1;
        let n_cols = self.dim.0;
        debug_assert_eq!(m.dim, self.dim);
        debug_assert_eq!(v_buf.dim, self.dim);
        debug_assert_eq!(u.len(), n_rows);
        debug_assert_eq!(v_vec.len(), n_cols);
        let u_data = u.as_slice();
        let v_data = v_vec.as_slice();
        let one_minus_b1 = 1.0 - beta1;
        let one_minus_b2 = 1.0 - beta2;
        for r in 0..n_rows {
            let u_r = u_data[r];
            let row_start = r * n_cols;
            for c in 0..n_cols {
                let idx = row_start + c;
                let g = u_r * v_data[c];
                let m_new = beta1 * m.data[idx] + one_minus_b1 * g;
                let v_new = beta2 * v_buf.data[idx] + one_minus_b2 * g * g;
                m.data[idx] = m_new;
                v_buf.data[idx] = v_new;
                let m_hat = m_new / bias_corr1;
                let v_hat = v_new / bias_corr2;
                self.data[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    // FIXME: Should be a macro
    fn print_internal_io<W: IoWrite>(&self, f: &mut W, tabbing: usize) -> std::io::Result<()> {
        let tab_str = "\t".repeat(tabbing);
        write!(f, "{}{}", tab_str, "[")?;
        for r in 0..self.dim.1
        {
            write!(f, "{}", "[")?;
            for c in 0..self.dim.0
            {
                write!(f, "{}", self[r][c])?;
                if c == self.dim.0 - 1
                {
                    write!(f, "{}", "]")?;
                    if r != self.dim.1 - 1
                    {
                        write!(f, "{}{}", "\n", tab_str)?;
                    }
                }
                else {
                    write!(f, "{}", " ")?;
                }
            }
        }
        write!(f, "{}", "]")?;
        Ok(())
    }

    // FIXME: Should be a macro
    fn print_internal_fmt<W: FmtWrite>(&self, f: &mut W, tabbing: usize) -> std::fmt::Result {
        let tab_str = "\t".repeat(tabbing);
        write!(f, "{}{}", tab_str, "[")?;
        for r in 0..self.dim.1
        {
            write!(f, "{}", "[")?;
            for c in 0..self.dim.0
            {
                write!(f, "{}", self[r][c])?;
                if c == self.dim.0 - 1
                {
                    write!(f, "{}", "]")?;
                    if r != self.dim.1 - 1
                    {
                        write!(f, "{}{}", "\n", tab_str)?;
                    }
                }
                else {
                    write!(f, "{}", " ")?;
                }
            }
        }
        write!(f, "{}", "]")?;
        Ok(())
    }
}

// column vector: 1 column, SIZE rows
impl FennVector{
    pub fn new(dim: usize) -> Self {
        FennVector {
            matrix: FennMatrix::new((dim, 1))
        }
    }
    pub fn ones(dim: usize) -> Self {
        FennVector {
            matrix: FennMatrix::ones((dim, 1))
        }
    }
    pub fn rand(dim: usize, min: f64, max: f64) -> Self {
        FennVector {
            matrix: FennMatrix::rand((dim, 1), min, max)
        }
    }

    pub fn len(&self) -> usize
        { self.matrix.dim.1 }

    pub fn from_slice<const SIZE: usize>(data: &[f64; SIZE]) -> Self {
        FennVector {
            matrix: FennMatrix::from_vec(data.to_vec(), (SIZE, 1))
        }
    }

    pub fn from_iter<I: Iterator<Item = f64>>(iter: I) -> Self
        { FennVector::from_vec(iter.collect()) }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        FennVector {
            matrix: FennMatrix::from_vec(data, (len, 1))
        }
    }

    pub fn move_to_matrix(self) -> FennMatrix
        { self.matrix }

    pub fn to_matrix(&self) -> FennMatrix
        { self.matrix.clone() }

    pub fn to_matrix_ref(&self) -> &FennMatrix
        { &self.matrix }

    pub fn dimensions(&self) -> (usize, usize)
        { self.matrix.dim }

    pub fn dot(&self, v: &FennVector) -> f64
    {
        assert_eq!(self.len(), v.len());
        self.zip(v, move |(x,y)| x * y).sum()
    }

    fn as_slice(&self) -> &[f64]
        { &self.matrix.data.as_slice() }

    pub fn cross(&self, v: &FennVector) -> FennMatrix
        { FennMatrix::new((v.matrix.dim.1, v.matrix.dim.1)) }

    pub fn map<F: Fn(&f64) -> f64>(&self, f: F) -> FennVector
        { self.matrix.map(f).to_vec().expect("This should never fail") }

    pub fn element_wise_mult(&self, other: &FennVector) -> FennVector
        { self.zip(other, move |(l, r)| l*r) }

    // In-place: self -= scale * other.
    pub fn sub_scaled(&mut self, scale: f64, other: &FennVector) {
        debug_assert_eq!(self.len(), other.len());
        let other_data = other.as_slice();
        for i in 0..self.matrix.data.len() {
            self.matrix.data[i] -= scale * other_data[i];
        }
    }

    // Fused Adam update on `self`. See FennMatrix::adam_update_outer for the math.
    pub fn adam_update(
        &mut self,
        m: &mut FennVector,
        v_buf: &mut FennVector,
        grad: &FennVector,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) {
        debug_assert_eq!(m.len(), self.len());
        debug_assert_eq!(v_buf.len(), self.len());
        debug_assert_eq!(grad.len(), self.len());
        let one_minus_b1 = 1.0 - beta1;
        let one_minus_b2 = 1.0 - beta2;
        let n = self.matrix.data.len();
        let grad_data = grad.as_slice();
        for i in 0..n {
            let g = grad_data[i];
            let m_new = beta1 * m.matrix.data[i] + one_minus_b1 * g;
            let v_new = beta2 * v_buf.matrix.data[i] + one_minus_b2 * g * g;
            m.matrix.data[i] = m_new;
            v_buf.matrix.data[i] = v_new;
            let m_hat = m_new / bias_corr1;
            let v_hat = v_new / bias_corr2;
            self.matrix.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    pub fn zip<F: FnMut((&f64, &f64)) -> f64>(&self, other: &FennVector, f: F) -> FennVector
    {
        assert_eq!(self.dimensions(), other.dimensions(), "Dimensions {:?} and {:?} are not equal", self.dimensions(), other.dimensions());
        FennVector::from_iter(self.as_slice()
                                  .iter()
                                  .zip(other.as_slice())
                                  .map(f))
    }

    pub fn sum(&self) -> f64
    { self.fold(0.0, move |l, r| l + r) }

    pub fn argmax(&self) -> usize {
        self.as_slice().iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn one_hot(&self) -> FennVector {
        let n = self.len();
        let idx = self.argmax();
        let mut data = vec![0.0; n];
        data[idx] = 1.0;
        FennVector::from_vec(data)
    }

    pub fn fold<F: Fn(f64, &f64) -> f64>(&self, init: f64, f: F) -> f64
        { self.as_slice().iter().fold(init, f) }

    pub fn abs(&self) -> FennVector
        { self.map(move |e| f64::abs(*e)) }

    pub fn pow(&self, exp: f64) -> FennVector
        { self.map(move |e| e.powf(exp)) }

    pub fn print(&self, tabbing: usize)
        { self.matrix.print(tabbing); }
}

// Useful Impls

impl std::fmt::Display for FennMatrix
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
        { self.print_internal_fmt(f, 0) }
}
impl std::fmt::Display for FennVector
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
        { self.matrix.print_internal_fmt(f, 0) }
}

impl std::ops::Index<usize> for FennMatrix
{
    type Output = [f64];

    fn index(&self, row_index: usize) -> &Self::Output {
        assert!(row_index < self.dim.1);
        let row_start = self.dim.0*row_index;
        &self.data[row_start..row_start+self.dim.0]
    }
}

impl std::ops::Index<usize> for FennVector
{
    type Output = f64;

    fn index(&self, row_index: usize) -> &Self::Output {
        assert!(row_index < self.matrix.dim.1);
        &self.matrix[row_index][0]
    }
}
impl std::ops::IndexMut<usize> for FennMatrix
{
    fn index_mut(&mut self, row_index: usize) -> &mut Self::Output {
        assert!(row_index < self.dim.1);
        let row_start = self.dim.0*row_index;
        &mut self.data[row_start..row_start+self.dim.0]
    }
}
impl std::ops::IndexMut<usize> for FennVector
{
    fn index_mut(&mut self, row_index: usize) -> &mut Self::Output {
        assert!(row_index < self.matrix.dim.1);
        &mut self.matrix[row_index][0]
    }
}

impl From<FennVector> for FennMatrix
{
    fn from(value: FennVector) -> Self
        { value.matrix }
}

impl TryFrom<FennMatrix> for FennVector
{
    type Error = Errors;
    fn try_from(value: FennMatrix) -> Result<Self, Self::Error>
        { value.to_vec() }
}

impl Mul for FennMatrix
{
    type Output = Self;
    fn mul(self, other: Self) -> Self
        { &self * &other }
}

impl Mul<&FennMatrix> for f64
{
    type Output = FennMatrix;
    fn mul(self, rhs: &FennMatrix) -> Self::Output
        { rhs.map(|e| e*self) }
}

impl Mul<FennMatrix> for f64
{
    type Output = FennMatrix;
    fn mul(self, rhs: FennMatrix) -> Self::Output
        { self * &rhs }
}

impl Mul for &FennMatrix
{
    type Output = FennMatrix;
    fn mul(self, other: &FennMatrix) -> Self::Output
    {
        debug_assert_eq!(self.dim.0, other.dim.1,
            "Matmul dim mismatch: lhs is {}x{}, rhs is {}x{}",
            self.dim.1, self.dim.0, other.dim.1, other.dim.0);

        let n_rows = self.dim.1;
        let n_inner = self.dim.0;
        let n_cols = other.dim.0;

        let mut m = FennMatrix {
            data: vec![0f64; n_cols * n_rows],
            dim: (n_cols, n_rows),
        };

        for i in 0..n_rows {
            let a_row_start = i * n_inner;
            let c_row_start = i * n_cols;
            for k in 0..n_inner {
                let a_ik = self.data[a_row_start + k];
                let b_row_start = k * n_cols;
                for j in 0..n_cols {
                    m.data[c_row_start + j] += a_ik * other.data[b_row_start + j];
                }
            }
        }
        m
    }
}

impl Add<FennVector> for FennVector
{
    type Output = FennVector;
    fn add(self, rhs: FennVector) -> Self::Output
        { &self + &rhs }
}

impl Add<FennVector> for &FennVector
{
    type Output = FennVector;
    fn add(self, rhs: FennVector) -> Self::Output
        { self + &rhs }
}

impl Add<FennVector> for FennMatrix
{
    type Output = FennVector;
    fn add(self, rhs: FennVector) -> Self::Output
        { &self + &rhs }
}

impl Add<&FennVector> for &FennVector
{
    type Output = FennVector;
    fn add(self, rhs: &FennVector) -> Self::Output
    {
        FennVector::from_vec(
            self.as_slice()
                      .iter()
                      .zip(rhs.as_slice())
                      .map(|(l, r)| l + r)
                      .collect())
    }
}

impl Add<&FennVector> for FennMatrix
{
    type Output = FennVector;
    fn add(self, rhs: &FennVector) -> Self::Output
        { &self + rhs }
}

impl Add<FennVector> for &FennMatrix
{
    type Output = FennVector;
    fn add(self, rhs: FennVector) -> Self::Output
        { self + &rhs }
}

impl Add<&FennVector> for &FennMatrix
{
    type Output = FennVector;
    fn add(self, rhs: &FennVector) -> Self::Output
    {
        debug_assert_eq!(self.dim.0, 1, "Matrix must be a column vector to add with FennVector");
        debug_assert_eq!(self.dim.1, rhs.len());
        FennVector::from_vec(
            self.data.iter()
                .zip(rhs.as_slice())
                .map(|(l, r)| l + r)
                .collect())
    }
}

impl Sub<Self> for FennVector
{
    type Output = FennVector;
    fn sub(self, rhs: FennVector) -> Self::Output
        { (&self) - (&rhs) }
}

impl Sub<FennVector> for &FennVector
{
    type Output = FennVector;
    fn sub(self, rhs: FennVector) -> Self::Output
        { (self) - (&rhs) }
}

impl Sub<&FennVector> for FennVector
{
    type Output = FennVector;
    fn sub(self, rhs: &FennVector) -> Self::Output
        { (&self) - (rhs) }
}

impl Sub<Self> for &FennVector
{
    type Output = FennVector;
    fn sub(self, rhs: &FennVector) -> Self::Output
    {
        FennVector::from_vec(
            self.as_slice()
                      .iter()
                      .zip(rhs.as_slice())
                      .map(|(l, r)| l - r)
                      .collect())
    }
}

impl Sub<Self> for FennMatrix
{
    type Output = FennMatrix;
    fn sub(self, rhs: FennMatrix) -> Self::Output
        { (&self) - (&rhs) }
}

impl Sub<FennMatrix> for &FennMatrix
{
    type Output = FennMatrix;
    fn sub(self, rhs: FennMatrix) -> Self::Output
        { (self) - (&rhs) }
}

impl Sub<&FennMatrix> for FennMatrix
{
    type Output = FennMatrix;
    fn sub(self, rhs: &FennMatrix) -> Self::Output
        { (&self) - (rhs) }
}

impl Sub<Self> for &FennMatrix
{
    type Output = FennMatrix;
    fn sub(self, rhs: &FennMatrix) -> Self::Output
    {
        FennMatrix::from_vec(
            self.data.as_slice()
                      .iter()
                      .zip(rhs.data.as_slice())
                      .map(|(l, r)| l - r)
                      .collect(), (self.dim.1, self.dim.0))
    }
}

impl Mul<&FennVector> for f64
{
    type Output = FennVector;
    fn mul(self, rhs: &FennVector) -> Self::Output
        { rhs.map(|e| e*self) }
}

impl Mul<FennVector> for f64
{
    type Output = FennVector;
    fn mul(self, rhs: FennVector) -> Self::Output
        { self * &rhs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_zero_matrix3x3()
    {
        let m = FennMatrix::new((2, 3));
        assert_eq!(m[0][0], 0f64);
        assert_eq!(m[0][1], 0f64);
        assert_eq!(m[0][2], 0f64);
        assert_eq!(m[1][0], 0f64);
        assert_eq!(m[1][1], 0f64);
        assert_eq!(m[1][2], 0f64);
    }
    #[test]
    fn create_ones_matrix3x3()
    {
        let m = FennMatrix::from_slice(&[[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(m[0][0], 1f64);
        assert_eq!(m[0][1], 2f64);
        assert_eq!(m[0][2], 3f64);
        assert_eq!(m[1][0], 4f64);
        assert_eq!(m[1][1], 5f64);
        assert_eq!(m[1][2], 6f64);
    }

    #[test]
    fn multiply_1x3_3x1_1x1()
    {
        let m2 = FennMatrix::from_slice(&[[1., 2., 3.]]);
        let m = FennMatrix::from_slice(&[[1.], [2.], [3.]]);
        let m3 = m2 * m;
        assert_eq!(m3.dim, (1, 1));
        assert_eq!(m3[0][0], 14.0);
    }

    #[test]
    fn multiply_3x3_3x3_3x3()
    {
        let m2 = FennMatrix::from_slice(&[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
        let m = FennMatrix::from_slice(&[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]);
        let m3 = m2 * m;
        assert_eq!(m3.dimensions(), (3, 3));
        assert_eq!(m3[0][0], 6f64);
        assert_eq!(m3[0][1], 12f64);
        assert_eq!(m3[0][2], 18f64);
        assert_eq!(m3[1][0], 6f64);
        assert_eq!(m3[1][1], 12f64);
        assert_eq!(m3[1][2], 18f64);
        assert_eq!(m3[2][0], 6f64);
        assert_eq!(m3[2][1], 12f64);
        assert_eq!(m3[2][2], 18f64);
        //m3.print(0);
    }

    #[test]
    fn mutate_matrix3x3()
    {
        let mut m = FennMatrix::from_slice(&[[1., 2., 3.], [4., 5., 6.]]);
        m[0][0] = 64f64;
        m[1][1] = 3434f64;
        m[1][2] = 0.9999f64;

        assert_eq!(m[0][0], 64f64);
        assert_eq!(m[0][1], 2f64);
        assert_eq!(m[0][2], 3f64);
        assert_eq!(m[1][0], 4f64);
        assert_eq!(m[1][1], 3434f64);
        assert_eq!(m[1][2], 0.9999f64);
    }

    fn create_zero_vec3x1()
    {
        let v = FennVector::new(3);
        assert_eq!(v[0], 0f64);
        assert_eq!(v[1], 0f64);
        assert_eq!(v[2], 0f64);
    }
    #[test]
    fn create_ones_vec3x1()
    {
        let v = FennVector::from_slice(&[1., 2., 3.]);
        assert_eq!(v[0], 1f64);
        assert_eq!(v[1], 2f64);
        assert_eq!(v[2], 3f64);
    }
    #[test]
    fn mutate_vec3x1()
    {
        let mut v = FennVector::from_slice(&[1., 2., 3.]);
        v[0] = 64f64;
        v[1] = 3434f64;
        v[2] = 0.9999f64;

        assert_eq!(v[0], 64f64);
        assert_eq!(v[1], 3434f64);
        assert_eq!(v[2], 0.9999f64);
    }

    #[test]
    fn print()
    {
        let m = FennMatrix::rand((2, 2), 0., 1.);
        let result = std::panic::catch_unwind(|| m.print_internal_io(&mut sink(), 0));
        assert!(result.is_ok());
    }
}
