
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms

pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)


pq.set_left_operators(['e1(m,e)'])
pq.set_right_operators(['1'])

pq.add_st_operator(1.0,['f'],['t1'])
pq.add_st_operator(1.0,['v'],['t1'])

pq.add_st_operator(1.0,['f','t2'],['t1'])
pq.add_st_operator(1.0,['v','t2'],['t1'])

pq.add_st_operator(-1.0,['t2','f'],['t1'])
pq.add_st_operator(-1.0,['t2','v'],['t1'])

pq.simplify()
# grab list of fully-contracted strings, then print
single_amplitudes = pq.fully_contracted_strings()
for term in single_amplitudes:
    print(term)
print('-------------')
pq.clear()

pq.set_left_operators(['e2(m,n,f,e)'])
# T1-transformed H
pq.add_st_operator(1.0,['f'],['t1'])
pq.add_st_operator(1.0,['v'],['t1'])

# CC2
pq.add_commutator(1.0, ['f'], ['t2'])

# BEGIN CCSD
# [H, T2]
# pq.add_st_operator(1.0,['f','t2'],['t1'])
# pq.add_st_operator(1.0,['v','t2'],['t1'])

# pq.add_st_operator(-1.0,['t2','f'],['t1'])
# pq.add_st_operator(-1.0,['t2','v'],['t1'])

# 0.5 * [[H, T2], T2]
# = 0.5 * [H*T2 - T2*H, T2] = 0.5 * (H*T2*T2 -T2*H*T2 - T2*H*T2 + T2*T2*H)
# pq.add_st_operator(0.5,['f','t2','t2'],['t1'])
# pq.add_st_operator(0.5,['v','t2','t2'],['t1'])

# pq.add_st_operator(-1.0,['t2', 'f','t2'],['t1'])
# pq.add_st_operator(-1.0,['t2', 'v','t2'],['t1'])

# pq.add_st_operator(0.5,['t2','t2', 'f'],['t1'])
# pq.add_st_operator(0.5,['t2','t2', 'v'],['t1'])
# END CCSD

pq.simplify()
double_amplitudes = pq.fully_contracted_strings()
for term in double_amplitudes:
    print(term)
pq.clear()


# CC2 Jacobian

# doubles/doubles
pq.clear()
pq.set_left_operators(['e2(m,n,f,e)'])
pq.set_right_operators(['1'])
pq.add_commutator(1.0, ['f'], ['r2'])
pq.simplify()
jac_dd = pq.fully_contracted_strings()
jac_dd = contracted_strings_to_tensor_terms(jac_dd)

# singles/doubles
pq.clear()
pq.set_left_operators(['e1(m,e)'])
pq.set_right_operators(['1'])
pq.add_st_operator(1.0,['f', 'r2'], ['t1'])
pq.add_st_operator(1.0,['v', 'r2'], ['t1'])
pq.add_st_operator(-1.0,['r2', 'f'], ['t1'])
pq.add_st_operator(-1.0,['r2', 'v',], ['t1'])
pq.simplify()
jac_sd = pq.fully_contracted_strings()
jac_sd = contracted_strings_to_tensor_terms(jac_sd)

# doubles/singles
# pq.clear()
# pq.set_left_operators(['e2(m,n,f,e)'])
# pq.set_right_operators(['1'])
# pq.add_st_operator(1.0, ['f', 'r1'], ['t1'])
# pq.add_st_operator(1.0, ['v', 'r1'], ['t1'])
# pq.add_st_operator(-1.0,['r1', 'f'], ['t1'])
# pq.add_st_operator(-1.0,['r1', 'v',], ['t1'])
# pq.simplify()
# pq.print()
# jac_ds = pq.fully_contracted_strings()
# jac_ds = contracted_strings_to_tensor_terms(jac_ds)

# singles/singles
pq.clear()
pq.set_left_operators(['e1(m,e)'])
pq.set_right_operators(['1'])

pq.add_st_operator(1.0,['f', 'r1'], ['t1'])
pq.add_st_operator(1.0,['v', 'r1'], ['t1'])
pq.add_st_operator(-1.0,['r1', 'f'], ['t1'])
pq.add_st_operator(-1.0,['r1', 'v',], ['t1'])

# [[H, T2], r1]
# = [H*T2 - T2*H, r1] = (H*T2*r1 -T2*H*r1 - r1*H*T2 + r1*T2*H)
pq.add_st_operator(1.0, ['f','t2','r1'],['t1'])
pq.add_st_operator(1.0, ['v','t2','r1'],['t1'])

pq.add_st_operator(-1.0,['t2', 'f','r1'],['t1'])
pq.add_st_operator(-1.0,['t2', 'v','r1'],['t1'])

pq.add_st_operator(-1.0,['r1', 'f', 't2'],['t1'])
pq.add_st_operator(-1.0,['r1', 'v', 't2'],['t1'])

pq.add_st_operator(1.0, ['r1','t2', 'f'],['t1'])
pq.add_st_operator(1.0, ['r1','t2', 'v'],['t1'])

pq.simplify()
jac_ss = pq.fully_contracted_strings()
jac_ss = contracted_strings_to_tensor_terms(jac_ss)


print("CC2 gs amplitude equations")
tia_tensors = contracted_strings_to_tensor_terms(single_amplitudes)
for my_term in tia_tensors:
        print(my_term.einsum_string(update_val='t1new',
                                    optimize=False,
                                    output_variables=('m', 'e')))
        # print()
print("-----")
tijab_tensors = contracted_strings_to_tensor_terms(double_amplitudes)
for my_term in tijab_tensors:
        print(my_term.einsum_string(update_val='t2new',
                                    optimize=False,
                                    output_variables=('m', 'n', 'e', 'f')))
print("-----")

print("CC2 Jacobian matrix-vector product")
print("doubles-doubles")
for term in jac_dd:
    print(term.einsum_string(update_val='pphh',
                                optimize=False,
                                output_variables=('m', 'n', 'e', 'f')))

# print("doubles-singles")
# for term in jac_ds:
#     print(term.einsum_string(update_val='pphh',
#                                 optimize=False,
#                                 output_variables=('m', 'n', 'e', 'f')))

print("singles-doubles")
for term in jac_sd:
    print(term.einsum_string(update_val='ph',
                                optimize=False,
                                output_variables=('m', 'e')))
print("singles-singles")
for term in jac_ss:
    print(term.einsum_string(update_val='ph',
                                optimize=False,
                                output_variables=('m', 'e')))