use crate::general::kmeans_revisited::StatsError::EmptyObservation;
use num_traits::real::Real;
use rand::RngCore;
use std::array::from_fn;
use std::collections::HashSet;
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Sub};

/// For k-means to work best, we need to have a way to normalize values across different dimensions
/// This could for instance be the zscore function, returning the difference between a value and the mean of all values, divided by the standard deviation
/// (see: https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html )
trait Normalizable<Normalised> {
    fn normalise(self) -> Vec<Normalised>;
}

#[derive(Debug)]
pub struct Stats<Number> {
    pub count: Number,
    pub mean: Number,
    pub sum: Number,
    pub stddev: Number,
    pub observations: Vec<Number>,
}

pub enum StatsError {
    EmptyObservation,
}

pub trait HasStats<Number> {
    fn stats(self) -> Result<Stats<Number>, StatsError>;
}

impl<'i, I, T, Number> HasStats<Number> for I
where
    I: IntoIterator<Item = &'i T>,
    T: Into<Number> + Copy + 'i,
    Number: std::ops::Div<Output = Number>
        + Sub<Output = Number>
        + Add<Output = Number>
        + Mul<Number, Output = Number>
        + Copy
        + Sum
        + Real,
{
    fn stats(self) -> Result<Stats<Number>, StatsError> {
        let mut sum = None;
        let numbers: Vec<Number> = self
            .into_iter()
            .map(|item| {
                let num: Number = (*item).into();
                sum = match sum {
                    None => Some(num),
                    Some(old) => Some(old + num),
                };
                num
            })
            .collect();
        if numbers.is_empty() {
            return Err(EmptyObservation);
        }
        let sum = sum.unwrap();
        let len = Number::from(numbers.len()).unwrap();
        let mean = sum / len;
        let variance = numbers
            .iter()
            .map(|value| {
                let diff = mean - *value;
                diff * diff
            })
            .sum::<Number>()
            / len;
        let stddev = variance.sqrt();
        Ok(Stats {
            count: len,
            mean,
            sum,
            stddev,
            observations: numbers,
        })
    }
}

impl<'i, I, T, Number> Normalizable<Number> for I
where
    I: IntoIterator<Item = &'i T>,
    T: Into<Number> + Copy + 'i,
    Number: std::ops::Div<Output = Number>
        + Sub<Output = Number>
        + Add<Output = Number>
        + Mul<Number, Output = Number>
        + Copy
        + Sum
        + Real,
{
    fn normalise(self) -> Vec<Number> {
        match self.stats() {
            Err(_) => vec![],
            Ok(Stats {
                mean,
                stddev,
                observations,
                ..
            }) => observations
                .iter()
                .map(|value| (*value - mean) / stddev)
                .collect(),
        }
    }
}

pub trait NDimensions<const N: usize, T> {
    fn dimensions(&self) -> [T; N];
    fn from_dimensions(dimensions: &[T; N]) -> Self;
}

pub trait Distance<Number> {
    fn distance(&self, other: &Self) -> Number;
}

#[derive(Debug, Clone, Default)]
pub struct Cluster<Point, const DIMENSIONS: usize, Number>
where
    Point: NDimensions<DIMENSIONS, Number> + Distance<Number> + Copy + Clone + Debug,
    Number: Copy + Real + AddAssign,
{
    pub centroid: Option<Point>, // the centroid, if it has been computed (and cluster isn't empty)
    pub points: Vec<Point>, // keeps a reference to the points so that we can compute the centroid
    pub refs: HashSet<usize>, // for the sake of comparing clusters without comparing points, but their index in the original array
    _n: PhantomData<Number>,
}

impl<const DIMENSIONS: usize, Point, Number> Cluster<Point, DIMENSIONS, Number>
where
    Point: NDimensions<DIMENSIONS, Number> + Distance<Number> + Copy + Clone + Debug,
    Number: Copy + Real + AddAssign,
{
    pub fn add_point(&mut self, point: Point, reference: usize) {
        self.points.push(point);
        self.refs.insert(reference);
    }

    pub fn compute_center(&mut self) {
        self.centroid = find_center(&self.points);
    }

    /// Keeps the centroid, but clears the points (no assignment)
    pub fn clear(&mut self) {
        self.points.clear();
        self.refs.clear();
    }
}

pub struct KMeans<const K: usize, const DIMENSIONS: usize, Point, Number>
where
    Point: NDimensions<DIMENSIONS, Number> + Distance<Number> + Copy + Clone + Debug,
    Number: Copy + Real + AddAssign,
{
    clusters: [Cluster<Point, DIMENSIONS, Number>; K],
    points: Vec<Point>,
    max_iterations: usize,
    _n: PhantomData<Number>,
}

pub fn find_center<Point, Number, const DIMENSIONS: usize>(points: &[Point]) -> Option<Point>
where
    Point: NDimensions<DIMENSIONS, Number>,
    Number: Copy + Real + AddAssign,
{
    if points.is_empty() {
        return None;
    }
    let len = Number::from(points.len()).unwrap();
    // the center of a cluster of points is simply the mean across every dimension
    // one way to do this can be to transpose the points dimensions "vertically", and
    let points = points.iter().map(|p| p.dimensions()).collect::<Vec<_>>();
    let mut iter = points.into_iter();
    let center = &mut iter.next().unwrap();
    for dimensions in iter {
        for i in 0..DIMENSIONS {
            center[i] += dimensions[i];
        }
    }
    let dimensions = &center.map(|sum| sum / len); // we could even compute a rolling mean here (but we're already O(len*DIMENSIONS))
    Some(Point::from_dimensions(dimensions))
}

impl<const K: usize, const DIMENSIONS: usize, Point, Number> KMeans<K, DIMENSIONS, Point, Number>
where
    Point: NDimensions<DIMENSIONS, Number> + Distance<Number> + Copy + Clone + Debug + Eq,
    Number: Copy + Real + AddAssign + Debug,
{
    /// This assumes the data set (the points) has been normalised over every dimensions, first
    pub fn new(points: Vec<Point>, max_iterations: usize) -> Self {
        // originally, we'll assign points to clusters randomly
        let mut rng = rand::thread_rng();
        let mut clusters: [Cluster<Point, DIMENSIONS, Number>; K] = from_fn(|_| Cluster {
            centroid: None,
            points: vec![],
            refs: Default::default(),
            _n: Default::default(),
        });
        for (i, point) in points.iter().enumerate() {
            let cluster_idx = (rng.next_u32() % K as u32) as usize;
            clusters[cluster_idx].add_point(*point, i);
        }
        // compute the centroid of every cluster
        for cluster in clusters.iter_mut() {
            cluster.compute_center();
        }
        Self {
            clusters,
            points,
            max_iterations,
            _n: Default::default(),
        }
    }

    pub fn kmeans(&mut self) -> [Cluster<Point, DIMENSIONS, Number>; K] {
        let mut iterations = 0;
        while iterations < self.max_iterations {
            println!("iteration # {iterations}");
            println!("clusters # {:?}", self.clusters);
            // 1. take a snapshot of the centroids before the iterations, so that we check if the clustering is stable
            let old_assignment: [HashSet<usize>; K] = self.clusters.clone().map(|c| c.refs);
            // 2. re-organise points: put each point in the cluster it is closest to
            let mut new_clusters: [Cluster<Point, DIMENSIONS, Number>; K] = self.clusters.clone();
            // 2.1. clear assignments, but keep the centroid
            for cluster in new_clusters.iter_mut() {
                cluster.clear();
            }
            // 2.2.
            for (point_idx, point) in self.points.iter().enumerate() {
                let mut iter = self.clusters.iter();
                let cluster = iter.next().unwrap();
                let mut min_dist = (
                    0,
                    cluster
                        .centroid
                        .expect("NYI: empty cluster")
                        .distance(point),
                );
                for (idx, cluster) in iter.enumerate() {
                    let dist = cluster.centroid.expect("NYI empty cluster").distance(point);
                    if dist < min_dist.1 {
                        min_dist = (idx, dist);
                    }
                }
                // put the point into the cluster it is closest to
                new_clusters[min_dist.0].add_point(*point, point_idx);
            }
            // 3. is our new assignment stable? (i.e. the same as the previous one)
            let new_assignment = new_clusters.clone().map(|c| c.refs);
            if new_assignment == old_assignment {
                return new_clusters;
            }
            // 4. It's not stable, continue -> re-compute the center of every cluster
            for cluster in new_clusters.iter_mut() {
                cluster.compute_center();
            }
            self.clusters = new_clusters;
            iterations += 1;
        }
        self.clusters.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Normalizable;
    use crate::general::kmeans_revisited::{find_center, Distance, KMeans, NDimensions};

    fn assert_floats_eq(lhs: &f64, rhs: &f64) {
        assert!((lhs - rhs).abs() < f64::EPSILON, "{} != {}", lhs, rhs);
    }

    fn assert_floats_pair_eq((lhs, rhs): (&f64, &f64)) {
        assert_floats_eq(lhs, rhs)
    }

    fn assert_float_vecs_eq(lhs: Vec<f64>, rhs: Vec<f64>) {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter().zip(rhs.iter()).for_each(assert_floats_pair_eq)
    }

    #[test]
    fn test_normalise_some_vec() {
        let original = vec![3, 1, 6, 1, 5, 8, 1, 8, 10, 11];
        let normalised: Vec<f64> = original.normalise();
        println!("normalised = {:?}", normalised);
        assert_float_vecs_eq(
            normalised,
            vec![
                -0.6646185307460537,
                -1.2184673063677651,
                0.16615463268651331,
                -1.2184673063677651,
                -0.11076975512434237,
                0.7200034083082247,
                -1.2184673063677651,
                0.7200034083082247,
                1.2738521839299362,
                1.5507765717407918,
            ],
        );
    }

    #[test]
    fn test_normalise_some_slice() {
        let original = &[3, 1, 6, 1, 5, 8, 1, 8, 10, 11];
        let normalised_vec: Vec<f64> = original.to_vec().normalise();
        let normalised_slice: Vec<f64> = original.normalise();
        assert_float_vecs_eq(normalised_slice, normalised_vec)
    }

    #[derive(Debug, Copy, Clone)]
    struct Point {
        x: f64,
        y: f64,
    }

    impl PartialEq<Self> for Point {
        fn eq(&self, other: &Self) -> bool {
            self.distance(other) < f64::EPSILON // arbitrarily
        }
    }

    impl Eq for Point {}

    impl NDimensions<2, f64> for Point {
        fn dimensions(&self) -> [f64; 2] {
            [self.x, self.y]
        }

        fn from_dimensions(dimensions: &[f64; 2]) -> Self {
            Point {
                x: dimensions[0],
                y: dimensions[1],
            }
        }
    }

    impl Distance<f64> for Point {
        fn distance(&self, other: &Self) -> f64 {
            ((self.x - other.x).abs().powf(2.0) + (self.y - other.y).abs().powf(2.0)).sqrt()
        }
    }

    #[test]
    fn test_center() {
        // |  x   x
        // |    -
        // |  x   x
        let points = vec![
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 2.0 },
            Point { x: 1.0, y: 2.0 },
            Point { x: 2.0, y: 1.0 },
        ];
        let Point { x, y } =
            find_center(&points).expect("can compute the center of 4 2-dimensions points");
        assert_floats_eq(&x, &1.5);
        assert_floats_eq(&y, &1.5);
    }

    #[test]
    fn test_create_arbitrary_cluster() {
        let points = vec![
            // C1:  "close to 0,0"
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.1, y: -0.1 },
            Point { x: 0.15, y: 0.1 },
            Point { x: 0.05, y: -0.0 },
            Point { x: 0.1, y: 0.0 },
            Point { x: 0.17, y: -0.05 },
            // C2: "close to 1,1"
            Point { x: 1.2, y: 1.3 },
            Point { x: 0.95, y: 1.05 },
            Point { x: 1.1, y: 0.97 },
            Point { x: 0.96, y: 1.0 },
            // C3: "close to -1,-1"
            Point { x: -1.02, y: -0.9 },
            Point { x: -0.87, y: -1.0 },
            Point { x: -1.1, y: -1.0 },
            Point { x: -0.92, y: -0.98 },
            Point { x: -1.1, y: -0.96 },
        ];
        // not really normalised, will it matter?
        let mut kmeans: KMeans<3, 2, Point, f64> = KMeans::new(points, 100);
        let clusters = kmeans.kmeans();
        println!("clusters: {clusters:?}");
    }
}
