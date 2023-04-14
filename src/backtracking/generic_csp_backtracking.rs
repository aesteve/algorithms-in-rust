use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::io;
use std::io::Write;

/// This module is a introduction to Constraint Satisfaction Problems solving, using Backtracking
/// Constraint Satisfaction problems can be expressed by the following set of definitions:
/// 1. Our variables
/// 2. Domains for these variables (i.e. the list of values every variable can take)
/// 3. Constraints a set of values for variables need to satisfy

/// A Constraint checks if a list of assignments (variable to domain) is valid
pub trait Constraint<Assignments> {
    fn is_satisfied(&self, assignments: &Assignments) -> bool;
}

/// Then the most intuitive solver we can think of: walk through every possible assignment of a variable to a domain, and check if constraints are satisfied
/// If so, move on with this assignment, and attach a new variable to a value, if not, backtrack
pub struct CSPSolver<Variable, Domain, C: Constraint<HashMap<Variable, Domain>>>
where
    Variable: Copy + Clone + Eq + Hash,
    Domain: Copy + Clone + Eq,
{
    constraints: Vec<C>,
    variables: Vec<Variable>,
    domains: HashMap<Variable, Vec<Domain>>,
}

impl<Variable, Domain, C: Constraint<HashMap<Variable, Domain>>> CSPSolver<Variable, Domain, C>
where
    Variable: Copy + Clone + Eq + Hash + Debug,
    Domain: Copy + Clone + Eq + Debug,
{
    pub fn solve(&self) -> Option<HashMap<Variable, Domain>> {
        self.solve_backtrack(HashMap::with_capacity(self.variables.len()))
    }

    fn solve_backtrack(
        &self,
        assignment: HashMap<Variable, Domain>,
    ) -> Option<HashMap<Variable, Domain>> {
        // base-case, all assignments have been made and are valid
        if assignment.len() == self.variables.len() {
            return Some(assignment);
        }
        // Which variables are left unassigned?
        let unassigned_variable = self
            .variables
            .iter()
            .find(|v| !assignment.contains_key(*v))
            .unwrap();
        // Try to assign every possible value to that variable, and backtrack whenever we find an inconsistent state
        for value in self.domains.get(unassigned_variable).unwrap() {
            let mut new_assignment = assignment.clone(); // checkpoint
            new_assignment.insert(*unassigned_variable, *value);
            if self.is_consistent(&new_assignment) {
                // we can move on, bubble up the result as soon as we find one
                if let Some(result) = self.solve_backtrack(new_assignment) {
                    return Some(result);
                }
                // otherwise keep looking
            }
        }
        None
    }

    fn is_consistent(&self, assignment: &HashMap<Variable, Domain>) -> bool {
        self.constraints
            .iter()
            .all(|constraint| constraint.is_satisfied(assignment))
    }
}

#[cfg(test)]
mod tests {
    use crate::backtracking::generic_csp_backtracking::{CSPSolver, Constraint};
    use std::collections::{HashMap, HashSet};
    use std::io;
    use std::io::Write;

    /// A classical constraint-solving problem as example:
    /// Attaching a single digit to each alphabetic character {S, E, N, D, M, O, R, Y} so that we properly solve the equation:
    /// SEND+MORE=MONEY
    /// Here, the variables are our letters {S, E, N, D, M, O, R, Y}.
    /// For each variable, its domain is the range 0..=9
    /// What are the constraints? We only have one: checking that SEND+MORE=MONEY
    ///
    /// Let's check that our Brute-Force CSP solver works
    #[test]
    #[ignore] // run it manually if need be, but this is a bit too long to run on every build
    fn solve_send_more_money() {
        struct SendMoreMoney;
        impl Constraint<HashMap<char, u64>> for SendMoreMoney {
            fn is_satisfied(&self, assignments: &HashMap<char, u64>) -> bool {
                let unique_assignments = assignments.values().collect::<HashSet<_>>();
                if unique_assignments.len() < assignments.len() {
                    // we have duplicates, wrong assignment
                    return false;
                }
                let s = match assignments.get(&'s') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let e = match assignments.get(&'e') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let n = match assignments.get(&'n') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let d = match assignments.get(&'d') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let m = match assignments.get(&'m') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let o = match assignments.get(&'o') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let r = match assignments.get(&'r') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                let y = match assignments.get(&'y') {
                    None => return true, // incomplete assignment, move on
                    Some(assigned) => *assigned,
                };
                // first check that we have no duplicate assignment
                let unique_assignments = HashSet::from([s, e, n, d, m, o, r, y]);
                if unique_assignments.len() < 8 {
                    // we have duplicate values
                    return false;
                }
                //  SEND
                //  MORE
                // ______
                // MONEY
                let send = s * 1_000 + e * 100 + n * 10 + d;
                let more = m * 1_000 + o * 100 + r * 10 + e;
                let money = m * 10_000 + o * 1_000 + n * 100 + e * 10 + y;
                money == send + more
            }
        }

        let variables = vec!['s', 'e', 'n', 'd', 'm', 'o', 'r', 'y'];
        let mut domains: HashMap<char, Vec<u64>> = variables
            .iter()
            .map(|&char| (char, (0..=9).collect()))
            .collect();
        domains.insert('m', vec![1]); // to avoid starting with 0
        io::stdout().flush().unwrap();
        let solver = CSPSolver {
            constraints: vec![SendMoreMoney {}],
            variables,
            domains,
        };
        println!("before solve");
        io::stdout().flush().unwrap();
        let solution = solver.solve();
        assert!(solution.is_some());
        println!("solution = {solution:?}");
    }
}
