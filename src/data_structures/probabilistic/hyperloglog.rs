use num_traits::Pow;
use std::cmp::max;
use std::collections::hash_map::{DefaultHasher, RandomState};
use std::hash::{BuildHasher, Hash, Hasher};

/// An HyperLogLog is a probabilistic Data Structure aiming at computing an approximate count for some item
/// Counting items can be done in different ways, for example using a frequency map
/// But this requires to have use space proportional to the cardinality of the set we are counting items from
/// This can potentially become an issue when having a very huge cardinality (or a potentially unbounded stream of items as input)
/// In an HyperLogLog we will trade-off the exactitude of count for far less space complexity
/// Let's first define the operations allowed for such a data structure
trait CountUniqueApprox {
    /// Adds an item to count
    fn add<Item: Hash>(&mut self, item: Item);
    /// Return an approximated value of the count of distinct items previously seen
    fn count(&self) -> usize;
}

/// Let's dive in the implementation
/// I strongly suggest reading the definition https://en.wikipedia.org/wiki/HyperLogLog
/// After having read some more informal introduction like this one: https://towardsdatascience.com/hyperloglog-a-simple-but-powerful-algorithm-for-data-scientists-aed50fe47869
///
///
/// The idea behind is to rely on probabilities
/// Let's hash every element from the set and watch its binary representation (let's consider 3 bits only in the hash for simplicity)
/// We have 2^3 possible hash results => 8 (000, 001, 010, 011, 100, 101, 110, 111)
/// Let's consider ρ the position of the right foremost bit (or number of trailing zeros)
/// What are the odds having ρ=0
///     Every number ending with a 1 (xx1) would match this criteria: 111, 101, 011, 001: 4 / 8 = 1 / 2
/// What are the odds for ρ=1?
///     Every number in the form: x10 will match this condition: 010, 110. So 2 / 8 = 1 / 4
/// And finally, for ρ=2?
///     The only number is 100, meaning 1/8
///
/// We can generalize this to numbers above 3:
///     Probability(ρ=k) = 2^(-k-1)
///     (half the numbers have the right foremost bit at position 0)
/// As we observe elements in the incoming data stream (or set), let's hash them and count the MAXIMUM number of trailing zeros we have seen so far
/// This can give us a (very rough) idea of how many distinct values we have observed, if R is the maximum number of trailing zeros we've seen, then:
///     count_distinct ~= 2^R
///
/// This is the idea behind Flajolet-Martin's algorithm, although in reality statistical studies show that it's not exactly the right formula.
/// Therefore Flajolet-Martin added a corrective factor of 0.77351
///
const CORRECTIVE_FACTOR_PHI: f64 = 0.77351;
#[derive(Default)]
struct FlajoletMartinMax {
    max_trailing_zeros: u32,
}

impl CountUniqueApprox for FlajoletMartinMax {
    fn add<Item: Hash>(&mut self, item: Item) {
        let mut hasher = DefaultHasher::default(); // so that we always obtain the same hash for the same item
        item.hash(&mut hasher);
        let h = hasher.finish();
        self.max_trailing_zeros = max(self.max_trailing_zeros, h.trailing_zeros())
    }

    fn count(&self) -> usize {
        if self.max_trailing_zeros == 0 {
            return 0;
        }
        let r = self.max_trailing_zeros + 1;
        (2_f64.pow(r as f64) / CORRECTIVE_FACTOR_PHI).round() as usize
    }
}

/// The variant given in the Wikipedia article. Using a bitmap to keep the maximum of trailing zeros
/// Instead of computing the max, we're using a bitmap
#[derive(Default)]
struct FlajoletMartinBitmap {
    bitmap: u64,
}

impl CountUniqueApprox for FlajoletMartinBitmap {
    fn add<Item: Hash>(&mut self, item: Item) {
        let mut hasher = DefaultHasher::default(); // so that we always obtain the same hash for the same item
        item.hash(&mut hasher);
        let h = hasher.finish();
        self.bitmap |= 1 << h.trailing_zeros();
    }

    fn count(&self) -> usize {
        if self.bitmap == 0 {
            return 0;
        }
        let r = self.bitmap.trailing_ones();
        (2_f64.pow(r as f64) / CORRECTIVE_FACTOR_PHI).round() as usize
    }
}

/// Thing is: this implementation sounds really sensitive to outliers
/// If, by any bad luck, the first item we observe in the data stream is 10000000, we'll be quite far off
/// How can we improve that sensitivity?
/// The most immediate thing we can think of would be, in the same fashion as count-min sketch or bloom filters, to use multiple hash functions (M)
/// But then, how do we average the results?
/// Using the mean wouldn't be really good, since we'd still be very sensitive to outliers
/// Using the median could be better, but still, we'll be sensitive to outliers
/// This is a common statistical pattern, which actually gets solved by using:
///     * K groups of size L (L*K=M)
///     * compute the mean of each group
///     * return the median of the groups
#[derive(Debug)]
struct FlajoletMartinMultiMax<const K: usize, const L: usize> {
    bitmaps: [[u32; L]; K],
    hash_builders: Vec<RandomState>,
}

impl<const K: usize, const L: usize> Default for FlajoletMartinMultiMax<K, L> {
    fn default() -> Self {
        Self {
            bitmaps: [[0; L]; K],
            hash_builders: (0..K * L).map(|_| RandomState::new()).collect(),
        }
    }
}

fn mean_bitmaps<const L: usize>(items: [u32; L]) -> f64 {
    items.iter().map(|i| i.trailing_ones()).sum::<u32>() as f64 / L as f64
}

fn median(values: Vec<f64>) -> f64 {
    let mut values = values;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

impl<const K: usize, const L: usize> CountUniqueApprox for FlajoletMartinMultiMax<K, L> {
    fn add<Item: Hash>(&mut self, item: Item) {
        for (n, state) in self.hash_builders.iter_mut().enumerate() {
            let mut hasher = state.build_hasher();
            item.hash(&mut hasher);
            let h = hasher.finish();
            let i = n / L;
            let j = n % L;
            self.bitmaps[i][j] |= 1 << h.trailing_zeros();
        }
    }

    fn count(&self) -> usize {
        // use the mean
        let means: Vec<f64> = self.bitmaps.into_iter().map(mean_bitmaps).collect();
        let median_r = median(means);
        (2_f64.pow(median_r) / CORRECTIVE_FACTOR_PHI).round() as usize
    }
}

/// We did better (see the tests below)
/// But we used many hash functions unfortunately, when all we need is to "average out" or "just shuffle a bit"
/// A cleaver idea from Flajolet and Durand was to re-use the same hash function, and "cutting it into pieces"
/// Say we have a hash like so: 010100010
/// Instead of using 4 hash functions, we could use the first 2 bits in the hash, to indicate the index in which bucket we'll put the result in
/// As what we'll keep in every bucket, we'll just use the position of the left foremost 1 (no matter if we use leading 0s or trailing 0s, it doesn't change anything)
/// This led them to `LogLog` implementation
#[derive(Debug)]
struct LogLog {
    registers: Vec<u32>,
    address_bits: usize,
}

fn split_u64(h: u64, n: usize) -> (u64, u64) {
    let msb = h >> (64 - n);
    let lsb = (h << n) >> n;
    (msb, lsb)
}

impl LogLog {
    fn new(register_count: usize) -> Self {
        if !register_count.is_power_of_two() {
            panic!("The number of registers must be a power of two");
        }
        let address_bits = register_count.ilog2() as usize;
        Self {
            registers: vec![0; register_count],
            address_bits,
        }
    }

    fn split(&self, h: u64) -> (u64, u64) {
        split_u64(h, self.address_bits)
    }

}

fn alpha(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

impl CountUniqueApprox for LogLog {

    fn add<Item: Hash>(&mut self, item: Item) {
        let mut hasher = DefaultHasher::default(); // so that we always obtain the same hash for the same item
        item.hash(&mut hasher);
        let h = hasher.finish();
        let (address, rest) = self.split(h);
        let address = address as usize; // we have truncated address to a few bits (address_bits)
        let leading_zeros = rest.leading_zeros() - self.address_bits as u32;
        self.registers[address] = max(self.registers[address], leading_zeros)
    }

    fn count(&self) -> usize {
        // From the research paper: https://algo.inria.fr/flajolet/Publications/DuFl03-LNCS.pdf
        let m = self.registers.len();
        let alpha_m = alpha(m);
        let m = m as f64;
        let r = self.registers.iter().sum::<u32>() as f64 / m;
        (alpha_m * m * 2.0.pow(r).round()) as usize
    }
}

/// Finally, in 2007, the count formula got improved to use the Harmonic mean instead of the formula we've just used
#[derive(Debug)]
struct HyperLogLog {
    registers: Vec<u32>,
    address_bits: usize,
}

impl HyperLogLog {
    fn new(register_count: usize) -> Self {
        if !register_count.is_power_of_two() {
            panic!("The number of registers must be a power of two");
        }
        let address_bits = register_count.ilog2() as usize;
        Self {
            registers: vec![0; register_count],
            address_bits,
        }
    }

    fn split(&self, h: u64) -> (u64, u64) {
        split_u64(h, self.address_bits)
    }

}

impl CountUniqueApprox for HyperLogLog {

    fn add<Item: Hash>(&mut self, item: Item) {
        let mut hasher = DefaultHasher::default(); // so that we always obtain the same hash for the same item
        item.hash(&mut hasher);
        let h = hasher.finish();
        let (address, rest) = self.split(h);
        let address = address as usize; // we have truncated address to a few bits (address_bits)
        let leading_zeros = rest.leading_zeros() - self.address_bits as u32;
        self.registers[address] = max(self.registers[address], leading_zeros)
    }

    fn count(&self) -> usize {
        // From the research paper: https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
        let indicator = self.registers
            .iter()
            .map(|&m_j| 2.0.pow(-(m_j as f64)))
            .sum::<f64>();
        let m= self.registers.len();
        let alpha_m = alpha(m);
        (alpha_m * m.pow(2) as f64 / indicator).round() as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::data_structures::probabilistic::hyperloglog::{CountUniqueApprox, FlajoletMartinBitmap, FlajoletMartinMax, FlajoletMartinMultiMax, HyperLogLog, LogLog, split_u64};
    use std::cmp::min;
    use std::collections::HashSet;
    use std::fs;

    fn sample_text() -> String {
        fs::read_to_string(format!("{}/samples/romeo_and_juliet.txt", env!("CARGO_MANIFEST_DIR")))
            .expect("Could not read sample file")
    }

    fn sample_words() -> Vec<String> {
        sample_text()
            .trim()
            .split(['\n', '"', '(', ')', '[', ']', '1', '2', '3', ';', '\'', ':', ' ', '.', ',', '?', '!', '-'])
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    fn unique_words_in_sample() -> usize {
        sample_words().iter().collect::<HashSet<_>>().len()
    }

    #[test]
    fn fable_unique_words_using_flajolet_martin() {
        let mut words = sample_words();
        let mut flajolet_martin_max = FlajoletMartinMax::default();
        let mut flajolet_martin_bitmap = FlajoletMartinBitmap::default();
        for word in &words {
            flajolet_martin_max.add(word.clone());
            flajolet_martin_bitmap.add(word);
        }
        // Both implementations give the same result
        assert_eq!(flajolet_martin_max.count(), flajolet_martin_bitmap.count());

        // Uncomment this to get the error rate
        // let unique = words.iter().collect::<HashSet<_>>().len();
        // let estimated = flajolet_martin_bitmap.count();
        // let error = estimated.abs_diff(unique) as f64 / min(estimated, unique) as f64;
        // println!("{error}"); // less than 15% error
    }

    #[test]
    fn fable_unique_words_using_flajolet_martin_and_multiple_hashes() {
        let words = sample_words();
        let unique = words.iter().collect::<HashSet<_>>().len();

        let mut flajolet_martin_max = FlajoletMartinMax::default(); // for the sake of comparing implementations
        let mut flajolet_martin_10 = FlajoletMartinMultiMax::<5, 10>::default();
        for word in words {
            flajolet_martin_max.add(word.clone());
            flajolet_martin_10.add(word);
        }
        let estimated_single = flajolet_martin_max.count();
        let error_single =
            estimated_single.abs_diff(unique) as f64 / min(estimated_single, unique) as f64;
        let estimated_10 = flajolet_martin_10.count();
        let error_10 = estimated_10.abs_diff(unique) as f64 / min(estimated_single, unique) as f64;
        let improvement = error_single - error_10;
        // println!("improvement: {improvement}");
        assert!(improvement > 0.0); // hopefully we did better
    }

    #[test]
    fn fable_unique_words_using_loglog() {
        let mut loglog = LogLog::new(32); // 32 registers
        let mut flajolet_martin_10 = FlajoletMartinMultiMax::<5, 10>::default(); // re-using the previous implementation
        let words = sample_words();
        let unique = words.iter().collect::<HashSet<_>>().len();
        for word in words {
            loglog.add(word.clone());
            flajolet_martin_10.add(word);
        }
        let estimated_fla_durand = flajolet_martin_10.count();
        let error_fla_durand =
            estimated_fla_durand.abs_diff(unique) as f64 / min(estimated_fla_durand, unique) as f64;
        let estimated_loglog = loglog.count();
        let error_loglog = estimated_loglog.abs_diff(unique) as f64 / min(estimated_loglog, unique) as f64;
        let improvement = error_fla_durand - error_loglog;
        // println!("unique = {unique}");
        // println!("count Fla/Durand: {estimated_fla_durand}");
        // println!("count LogLog: {estimated_loglog}");
        // println!("improvement: {improvement}");
        assert!(improvement > -0.5); // hopefully we did better, but can't be entirely sure, let's make sure we don't perform way worse
    }

}
