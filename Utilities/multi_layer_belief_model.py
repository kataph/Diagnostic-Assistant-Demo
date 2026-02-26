import sys
import ast
from functools import reduce
from itertools import chain, combinations 
from .topology import Topology, minimal_dense_set
from .utils import get_key, get_set

class MultiLayerBeliefModel:
    
    """ This class defines an object Model containing all the information
        required to compute some degree of belief from a collection of uncertain pieces of evidence. 
        The user must provide as input the Justification frame (the criteria to consider an argument 
        good enough to justify degree of belief on those propositions implied by this argument); and 
        allocation fucntion (how we will interpret the sets of pieces of evience). For example, to 
        obtain the equivalent results to Demspter's rule of combination, the user must choose 
        'dempster_shafer' as Justification frame and 'set.intersection' as allocation function. """
    
   
    def __init__(self, S, E, J, f):
        """
        This is the constructor for Model.

        Args:
            S (set):          Space.
            E (dict):         Quantitative evidence.
            J (string):       Justification frame (dempster_shafer = dense and non-dense sets / strong_denseness =  dense sets of the topology). 
            f (function):     Allocation function (set.intersection/set.union/minimal_dense_set)

        Returns:
            None
        """
        
        self.space = S
        self.quantitative_evidence = E
        self.justification_frame = J
        self.allocation_function = f
        self.delta = None
        self.delta_tau = None
        self.delta_J = None
        

    
    
    def get_delta(self):
        """
        This method computes the delta function. For each combination of pieces of evidence,
        a corresponding delta value is computed. 
        
        Returns:
            dict (str, double) : For each combination of evidence, the delta value
        """
        
        # Avoid redundant computations
        if self.delta != None:
            return self.delta
        
        n = len(self.quantitative_evidence)
        pieces_of_evidence = [get_set(e) for e in self.quantitative_evidence.keys()]
        self.delta = {}
        
        for r in range(0,n+1):                               # For every size r of combination of pieces of evidence
            for combination in combinations(pieces_of_evidence, r):  # For all combinations of size r

                remaining_evidence = [e for e in pieces_of_evidence if e not in list(combination)]

                value = 1.0
                                
                for evidence in combination:  # For each piece of evidence in the combination
                    value *= self.quantitative_evidence[get_key(evidence)] # The value is multiplied by the new value
                                    
                for evidence in remaining_evidence: # For each piece of evidence not in the combination
                    value *= 1 - self.quantitative_evidence[get_key(evidence)] # The value is multiplied by 1 - the new value
                
                self.delta[str(list(combination))] = value   # We add the delta for this combination
        
        return self.delta
    
    def get_delta_tau(self):
        """
        This method computes the delta tau function. The allocation function is applied to each 
        argument of the delta function. The results constitutes the entries of the delta tau function.
        For each entry e, delta_tau(e) = sum(delta(a) if allocation(a) = e)
        
        Returns:
            dict(str, double) : For each combination of evidence, the delta value argument/ element of topology
        """

        # Avoid redundant computations
        if self.delta_tau != None:
            return self.delta_tau
        
        delta = self.get_delta().items()
        self.delta_tau = {}
        
        # Creation of new keys
        new_keys = []        
        for (combination,value) in delta: # Browsig entries of delta function
            argument = ast.literal_eval(combination) # Get the delta entry as a set

            if not argument:  # If delta entry is the empty set, new entry is S
                entry = self.space
            elif self.allocation_function == minimal_dense_set:
                entry = self.allocation_function(argument) # Compute allocation function on list
            else:
                entry = reduce(self.allocation_function, argument)  # Compute allocation function on list
            
            key = get_key(entry)


            if self.delta_tau.get(key):
                self.delta_tau[key] += value
            else:
                self.delta_tau[key] = value
            
            
            
        return self.delta_tau
    
    def get_justification_frame(self):
        """
        This method computes the justifications frame of the model.  
        
        Returns:
            list: list of sets
        """

        topology = Topology(self.quantitative_evidence, self.space)
        if self.justification_frame == 'dempster_shafer':
            justification_set = topology.get_topology()
            justification_set.remove(set())  # Remove the empty set from the topology
        elif self.justification_frame == 'strong_denseness':
            justification_set = topology.get_dense_sets()
        else:
            raise ValueError("Invalid value for J (Justification Frame). Must be 'dempster_shafer' or 'strong_denseness'.")
        
        return justification_set
    
    def get_normalization_factor(self):
        """
        This method computes the total sum of the frame of justifications.  
        
        Returns:
            denominator (double)
        """
        delta_tau = self.get_delta_tau()
        justification_set = self.get_justification_frame()
        
        denominator = sum(delta_tau[get_key(j)] for j in justification_set if get_key(j) in delta_tau) # Since delta_tau only contains elements with non-zero value, there may be j that are not in delta_tau
        
        return denominator

    
    def get_delta_J(self):
        """
        This method computes the deltaJ functions. 
        For each justification j in J, delta_J(j) = delta_tau(j) / sum(delta_tau(k) for k in J)
        
        Returns:
            dict (str, double) : For each combination of evidence, the delta value argument/ element of topology
        """

        # Avoid redundant computations
        if self.delta_J != None:
            return self.delta_J
        
    
        self.delta_J = {}
        delta_tau = self.get_delta_tau()
        """ topology = Topology(self.quantitative_evidence, self.space)
        if self.justification_frame == 'dempster_shafer':
            justification_set = topology.get_topology()
            justification_set.remove(set())  # Remove the empty set from the topology
        elif self.justification_frame == 'strong_denseness':
            justification_set = topology.get_dense_sets()
        else:
            raise ValueError("Invalid value for J (Justification Frame). Must be 'dempster_shafer' or 'strong_denseness'.")      
        
        
        denominator = sum(delta_tau[get_key(j)] for j in justification_set if get_key(j) in delta_tau) # Since delta_tau only contains elements with non-zero value, there may be j that are not in delta_tau
        print("Denominator: " + str(denominator)) """

        justification_set = self.get_justification_frame()
        denominator = self.get_normalization_factor()

        
        for justification in justification_set: # For all justification j
            key = get_key(justification)
            if key in delta_tau:                                    # If delta_tau(j) exists (has non-zero value)
                self.delta_J[key] = delta_tau[key] / denominator   #    Compute delta_j(j)
                
        testSum = sum(value for value in self.delta_J.values())
        print("Sum values: " + str(testSum))
        
        
        return self.delta_J
    

    def clear(self):
        """
        This method reinitializes the different tables of the objects.
        """

        self.delta = None
        self.delta_tau = None
        self.delta_J = None
    
    
    def degree_of_belief(self, proposition):
        """
        This method computes the degree of belief of a given proposition. 
        For a proposition p, degree_of_belief(p) = sum(delta_J(j) for j subset p)
        
        Returns:
            float : The degree of belief of p
        """
        
        delta_J = self.get_delta_J()
        
        return sum(delta_J[j] for j in delta_J if get_set(j).issubset(proposition))
    
    