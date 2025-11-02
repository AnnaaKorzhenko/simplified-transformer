import random
import itertools
import json
import csv
from typing import List, Tuple, Set
from collections import deque

class FormulaGenerator:
    """
    Generates temporal logic formulas in the form:
    Disjunction of Conjunctions of "not a until b and c"
    
    Parameters:
    - alphabet_size (pi): size of alphabet (or alphabet is provided)
    - num_disjunctions: number of disjunction clauses
    - num_conjunctions: number of conjunction clauses per disjunction
    - alphabet: optional list of alphabet symbols 
            
    """
    
    def __init__(self, alphabet_size: int, num_disjunctions: int, num_conjunctions: int, 
                 alphabet: List[str] = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.alphabet_size = alphabet_size
        self.num_disjunctions = num_disjunctions
        self.num_conjunctions = num_conjunctions
        
        if alphabet is not None:
            if len(alphabet) < alphabet_size:
                raise ValueError(f"Provided alphabet has {len(alphabet)} symbols but alphabet_size is {alphabet_size}")
            self.alphabet = alphabet[:alphabet_size]
        else:
            # Default: generate symbols like 'p1', 'p2', 'p3', ...
            self.alphabet = [f'p{i+1}' for i in range(alphabet_size)]
        
    def generate_formula(self) -> List[List[Tuple[str, str, str]]]:
        """
        Generate a formula: list of disjunctions, each containing conjunctions.
        Each conjunction is a tuple (a, b, c) representing "not a until b and c"
        
        Returns: List of lists of tuples
        """
        formula = []
        
        for _ in range(self.num_disjunctions):
            disjunction_clause = []
            for _ in range(self.num_conjunctions):
                # Randomly select a, b, c from alphabet
                a, b, c = random.sample(self.alphabet, 3)
                disjunction_clause.append((a, b, c))
            formula.append(disjunction_clause)
        
        return formula
    
    def formula_to_string(self, formula: List[List[Tuple[str, str, str]]]) -> str:
        """
        Convert formula to readable string representation
        """
        disj_parts = []
        for conj_clause in formula:
            conj_parts = []
            for a, b, c in conj_clause:
                conj_parts.append(f"(¬{a} U ({b} ∧ {c}))")
            disj_parts.append("(" + " ∧ ".join(conj_parts) + ")")
        return " ∨ ".join(disj_parts)


class FormulaEvaluator:
    """
    Evaluates whether a sequence satisfies a temporal logic formula
    Sequences are represented as lists of symbols (strings)
    """
    
    @staticmethod
    def evaluate_not_a_until_b_and_c(sequence: List[str], a: str, b: str, c: str) -> bool:
        """
        Evaluate "not a until b and c"
        This means: a does not hold until both b and c hold simultaneously
        
        Returns True if:
        - For all positions before the first occurrence where both b and c are true,
          a is false at those positions
        - If b and c never both occur, then a must never occur in the sequence
        
        Args:
            sequence: List of symbols (strings)
            a, b, c: Symbol strings (can be multi-character like 'p1', 'p2')
        """
        if not sequence:
            return True
        
        # Find first position where both b and c occur
        first_bc_position = None
        for i, symbol in enumerate(sequence):
            if symbol == b or symbol == c:
                # Check if have seen both b and c up to this point
                symbols_seen = set(sequence[:i+1])
                if b in symbols_seen and c in symbols_seen:
                    first_bc_position = i
                    break
        
        if first_bc_position is None:
            # b and c never both occur, so a must never occur
            return a not in sequence
        
        # Check that a does not occur before first_bc_position
        return a not in sequence[:first_bc_position]
    
    @staticmethod
    def evaluate_conjunction(sequence: List[str], conj_clause: List[Tuple[str, str, str]]) -> bool:
        """
        Evaluate a conjunction clause: all subformulas must be true
        """
        return all(
            FormulaEvaluator.evaluate_not_a_until_b_and_c(sequence, a, b, c)
            for a, b, c in conj_clause
        )
    
    @staticmethod
    def evaluate_formula(sequence: List[str], formula: List[List[Tuple[str, str, str]]]) -> bool:
        """
        Evaluate a disjunction of conjunctions formula
        Returns True if at least one disjunction clause is true
        """
        return any(
            FormulaEvaluator.evaluate_conjunction(sequence, conj_clause)
            for conj_clause in formula
        )


class SyntheticDataGenerator:
    """
    Generates synthetic sequence data with labels based on formula satisfaction
    Sequences are represented as lists of symbols (strings)
    """
    
    def __init__(self, alphabet: List[str], seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.alphabet = alphabet
    
    def generate_random_sequence(self, length: int) -> List[str]:
        """Generate a random sequence of given length as a list of symbols"""
        return random.choices(self.alphabet, k=length)
    
    def generate_sequence_satisfying_formula(
        self, 
        formula: List[List[Tuple[str, str, str]]], 
        length: int,
        max_attempts: int = 10000
    ) -> List[str]:
        """
        Generate a sequence that satisfies the formula
        Uses a constructive approach: pick one disjunction clause and satisfy it
        Returns a list of symbols
        """
        # Pick a random disjunction clause and construct sequence to satisfy it
        target_clause = random.choice(formula)
        sequence_list = [None] * length
        
        # Initialize with random symbols
        for i in range(length):
            sequence_list[i] = random.choice(self.alphabet)
        
        # For each (a, b, c) in target clause, ensure constraint: "not a until b and c"
        for a, b, c in target_clause:
            # Place b and c early (within first 2/3 of sequence)
            max_pos = max(1, (2 * length) // 3)
            pos_b = random.randint(0, max_pos - 1) if length > 1 else 0
            pos_c = random.randint(0, max_pos - 1) if length > 1 else 0
            
            # Ensure b and c are at different positions if possible
            if pos_b == pos_c and length > 1:
                pos_c = (pos_b + 1) % max_pos
            
            sequence_list[pos_b] = b
            sequence_list[pos_c] = c
            
            # Find the position where both b and c have appeared
            first_bc = max(pos_b, pos_c)
            
            # Ensure a does not appear before first_bc
            for i in range(first_bc):
                if sequence_list[i] == a:
                    # Replace with a random symbol that's not a
                    sequence_list[i] = random.choice([x for x in self.alphabet if x != a])
        
        # Fill any remaining None positions
        for i in range(length):
            if sequence_list[i] is None:
                sequence_list[i] = random.choice(self.alphabet)
        
        candidate = sequence_list
        
        # Verify it satisfies
        if FormulaEvaluator.evaluate_formula(candidate, formula):
            return candidate
        
        # If not working: try random with retries
        for _ in range(max_attempts):
            candidate = self.generate_random_sequence(length)
            if FormulaEvaluator.evaluate_formula(candidate, formula):
                return candidate
        
        # Return the constructed sequence 
        return candidate
    
    def generate_sequence_not_satisfying_formula(
        self,
        formula: List[List[Tuple[str, str, str]]],
        length: int,
        max_attempts: int = 10000
    ) -> List[str]:
        """
        Generate a sequence that does NOT satisfy the formula
        Strategy: violate at least one constraint in each disjunction clause
        Returns a list of symbols
        """
        # First try random sequences
        for _ in range(min(1000, max_attempts)):
            sequence = self.generate_random_sequence(length)
            if not FormulaEvaluator.evaluate_formula(sequence, formula):
                return sequence
        
        # If random does not work well, constructively violate constraints
        # To violate the formula, we need to violate ALL disjunction clauses
        sequence_list = [None] * length
        
        # For each disjunction clause, ensure it is violated
        for conj_clause in formula:
            # Violate at least one constraint in this clause
            a, b, c = random.choice(conj_clause)
            
            # Violation strategy: put a before both b and c have appeared
            # Place a early
            pos_a = random.randint(0, max(0, length // 3)) if length > 1 else 0
            sequence_list[pos_a] = a
            
            # Place b and c later (after a)
            if pos_a + 1 < length:
                pos_b = random.randint(pos_a + 1, length - 1)
                pos_c = random.randint(pos_a + 1, length - 1)
                
                if sequence_list[pos_b] is None:
                    sequence_list[pos_b] = b
                if sequence_list[pos_c] is None:
                    sequence_list[pos_c] = c
            else:
                # If length is very short, just place them
                if length > 0:
                    sequence_list[0] = a
                if length > 1:
                    sequence_list[1] = b
        
        # Fill remaining positions with random symbols
        for i in range(length):
            if sequence_list[i] is None:
                sequence_list[i] = random.choice(self.alphabet)
        
        candidate = sequence_list
        
        # Verify it does not satisfy
        if not FormulaEvaluator.evaluate_formula(candidate, formula):
            return candidate
        
        # If still satisfies, try more aggressive violation
        # Place a very early and delay b, c significantly
        sequence_list = [None] * length
        for conj_clause in formula:
            a, b, c = random.choice(conj_clause)
            sequence_list[0] = a
            if length > 2:
                sequence_list[length - 1] = b
                sequence_list[length - 2] = c
        
        for i in range(length):
            if sequence_list[i] is None:
                sequence_list[i] = random.choice(self.alphabet)
        
        candidate = sequence_list
        return candidate
    
    def generate_dataset(
        self,
        formula: List[List[Tuple[str, str, str]]],
        sequence_length: int,
        num_positive: int,
        num_negative: int,
        seed: int = None
    ) -> List[Tuple[List[str], int]]:
        """
        Generate a dataset of sequences with labels
        
        Returns: List of (sequence, label) tuples where:
            - sequence is a list of symbols (strings)
            - label is 1 (satisfies) or 0 (doesn't)
        """
        if seed is not None:
            random.seed(seed)
        
        dataset = []
        
        # Generate positive examples
        print(f"Generating {num_positive} positive examples...")
        for i in range(num_positive):
            seq = self.generate_sequence_satisfying_formula(formula, sequence_length)
            # Verify it actually satisfies
            if FormulaEvaluator.evaluate_formula(seq, formula):
                dataset.append((seq, 1))
            else:
                # Try a few more times
                for _ in range(10):
                    seq = self.generate_sequence_satisfying_formula(formula, sequence_length)
                    if FormulaEvaluator.evaluate_formula(seq, formula):
                        dataset.append((seq, 1))
                        break
                else:
                    dataset.append((seq, 0))  # Couldn't generate, mark as negative
        
        # Generate negative examples
        print(f"Generating {num_negative} negative examples...")
        for i in range(num_negative):
            seq = self.generate_sequence_not_satisfying_formula(formula, sequence_length)
            # Verify it actually doesn't satisfy
            if not FormulaEvaluator.evaluate_formula(seq, formula):
                dataset.append((seq, 0))
            else:
                # Try a few more times
                for _ in range(10):
                    seq = self.generate_sequence_not_satisfying_formula(formula, sequence_length)
                    if not FormulaEvaluator.evaluate_formula(seq, formula):
                        dataset.append((seq, 0))
                        break
                else:
                    dataset.append((seq, 1))  # Could not generate, mark as positive
        
        return dataset


def save_dataset_csv(dataset: List[Tuple[List[str], int]], filename: str, delimiter: str = ','):
    """
    Save dataset to CSV file
    Sequences are saved as delimiter-separated strings (e.g., 'p1,p2,p3')
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'label'])
        for seq, label in dataset:
            # Convert list of symbols to delimiter-separated string
            seq_str = delimiter.join(seq)
            writer.writerow([seq_str, label])


def load_dataset_csv(filename: str, delimiter: str = ',') -> List[Tuple[List[str], int]]:
    """Load dataset from CSV file"""
    dataset = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert delimiter-separated string back to list of symbols
            seq = row['sequence'].split(delimiter)
            dataset.append((seq, int(row['label'])))
    return dataset


def save_formula_json(formula: List[List[Tuple[str, str, str]]], filename: str, metadata: dict = None):
    """Save formula to JSON file"""
    data = {
        'formula': formula,
        'metadata': metadata or {}
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_formula_json(filename: str) -> Tuple[List[List[Tuple[str, str, str]]], dict]:
    """Load formula from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['formula'], data.get('metadata', {})


def generate_multiple_formulas_and_datasets(
    alphabet_size: int,
    num_disjunctions: int,
    num_conjunctions: int,
    sequence_length: int,
    num_positive: int,
    num_negative: int,
    num_formulas: int = 1,
    alphabet: List[str] = None,
    seed: int = None
) -> List[Tuple[List[List[Tuple[str, str, str]]], List[Tuple[List[str], int]]]]:
    """
    Generate multiple formulas and their corresponding datasets
    
    Returns: List of (formula, dataset) tuples
    """
    if seed is not None:
        random.seed(seed)
    
    results = []
    
    for i in range(num_formulas):
        print(f"\n{'='*60}")
        print(f"Generating formula {i+1}/{num_formulas}")
        print(f"{'='*60}")
        
        formula_gen = FormulaGenerator(
            alphabet_size=alphabet_size,
            num_disjunctions=num_disjunctions,
            num_conjunctions=num_conjunctions,
            alphabet=alphabet,
            seed=seed + i if seed is not None else None
        )
        
        formula = formula_gen.generate_formula()
        print(f"Formula: {formula_gen.formula_to_string(formula)}")
        
        data_gen = SyntheticDataGenerator(
            alphabet=formula_gen.alphabet,
            seed=seed + i if seed is not None else None
        )
        
        dataset = data_gen.generate_dataset(
            formula=formula,
            sequence_length=sequence_length,
            num_positive=num_positive,
            num_negative=num_negative,
            seed=seed + i if seed is not None else None
        )
        
        # Verify correctness
        correct = sum(
            1 for seq, label in dataset
            if FormulaEvaluator.evaluate_formula(seq, formula) == bool(label)
        )
        accuracy = correct / len(dataset) * 100
        print(f"Dataset accuracy: {accuracy:.2f}%")
        
        results.append((formula, dataset))
    
    return results


def main():
    """
    Example usage
    """
    # Parameters
    alphabet_size = 5  # pi
    num_disjunctions = 2
    num_conjunctions = 2
    sequence_length = 10
    num_positive = 50
    num_negative = 50
    
    print("=" * 60)
    print("Formula Generator")
    print("=" * 60)
    
    # Generate formula
    formula_gen = FormulaGenerator(
        alphabet_size=alphabet_size,
        num_disjunctions=num_disjunctions,
        num_conjunctions=num_conjunctions,
        seed=42
    )
    
    formula = formula_gen.generate_formula()
    print("\nGenerated Formula:")
    print(formula_gen.formula_to_string(formula))
    print("\nFormula structure:")
    for i, clause in enumerate(formula):
        print(f"  Disjunction {i+1}: {clause}")
    
    # Generate synthetic data
    print("\n" + "=" * 60)
    print("Generating Synthetic Data")
    print("=" * 60)
    
    data_gen = SyntheticDataGenerator(
        alphabet=formula_gen.alphabet,
        seed=42
    )
    
    dataset = data_gen.generate_dataset(
        formula=formula,
        sequence_length=sequence_length,
        num_positive=num_positive,
        num_negative=num_negative,
        seed=42
    )
    
    # Verify and display statistics
    positive_count = sum(1 for _, label in dataset if label == 1)
    negative_count = sum(1 for _, label in dataset if label == 0)
    
    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Positive (label 1): {positive_count}")
    print(f"  Negative (label 0): {negative_count}")
    
    # Verify correctness
    print("\nVerifying correctness...")
    correct = 0
    for seq, label in dataset:
        actual = FormulaEvaluator.evaluate_formula(seq, formula)
        expected = bool(label)
        if actual == expected:
            correct += 1
    
    accuracy = correct / len(dataset) * 100
    print(f"  Correct labels: {correct}/{len(dataset)} ({accuracy:.2f}%)")
    
    # Display sample sequences
    print("\nSample sequences:")
    print("  Positive examples (should satisfy formula):")
    pos_examples = [seq for seq, label in dataset if label == 1][:5]
    for seq in pos_examples:
        seq_str = ','.join(seq)
        print(f"    [{seq_str}] - {FormulaEvaluator.evaluate_formula(seq, formula)}")
    
    print("\n  Negative examples (should not satisfy formula):")
    neg_examples = [seq for seq, label in dataset if label == 0][:5]
    for seq in neg_examples:
        seq_str = ','.join(seq)
        print(f"    [{seq_str}] - {FormulaEvaluator.evaluate_formula(seq, formula)}")
    
    return formula, dataset


if __name__ == "__main__":
    main()

