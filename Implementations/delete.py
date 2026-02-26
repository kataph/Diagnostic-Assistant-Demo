
def pr(a,b):
    return f"""###  http://www.example.org/zorro/TestControls{a}_{b}
    <http://www.example.org/zorro/TestControls{a}_{b}> rdf:type owl:NamedIndividual ,
                                                        <http://www.example.org/zorro/Test> ;
                                            <http://www.example.org/zorro/hasTarget> <http://www.example.org/zorro/ControlSubsystem> ;
                                            <http://www.example.org/zorro/hasCost> 3 ;
                                            rdfs:comment "Detach the sequence of the control modules {a} to {b} from the circuit, then attach it to the power supply module on the left side and on the load module on the right side and check if the lamp turns on. " .
    """
    
def pr2(x):
    return (f"""###  http://www.example.org/zorro/DetatchedSwitch{x}
<http://www.example.org/zorro/DetatchedSwitch{x}> rdf:type owl:NamedIndividual ,
                                                          <http://www.example.org/zorro/Problem> ;
                                                 <http://www.example.org/zorro/hasTest> """ + ", ".join([f"<http://www.example.org/zorro/TestControls{a}_{b}>" for a in range(1,9) for b in range(1,9) if a<=x and x<=b]) + " ;\n" + 
            f"                                     rdfs:comment \"A cable is detached by one port of switch {x}\" .")

def pr3(x):
    return f"""###  http://www.example.org/zorro/Switch{x}
<http://www.example.org/zorro/Switch{x}> rdf:type owl:NamedIndividual ,
                                                <http://www.example.org/zorro/Component> ;
                                       <http://www.example.org/zorro/failsVia> <http://www.example.org/zorro/SwitchInternalFault> ;
                                       <http://www.example.org/zorro/failsVia> <http://www.example.org/zorro/DetatchedSwitch{x}> ;
                                       <http://www.example.org/zorro/hasFunction> <http://www.example.org/zorro/ControlCurrentFlow> ;
                                       <http://www.example.org/zorro/hasSuperComponent> <http://www.example.org/zorro/ControlModule{x}> .
"""

# for a in range(1,8+1):
#     for b in range(1, 8+1):
#         if a <= b:
#             print(pr(a,b))
            
for x in range(1,8+1):
    print(pr3(x))