"""Geometria de SEV Schlumberger, com unidades SI e metadados explícitos.
ASTM D6431: geometria e boas práticas; faixa numérica abaixo é escolha de pesquisa,
não um espaçamento obrigatório da norma.
"""
import numpy as np

def survey_geometry(n=20, min_ab2=1.0, max_ab2=200.0):
    ab2=np.geomspace(min_ab2,max_ab2,n)
    # AB > 5 MN; mudar o MN durante a expansão mantém tensão mensurável.
    mn=np.where(ab2<10,0.20,np.where(ab2<50,1.0,5.0))
    assert np.all(2*ab2>5*mn)
    k=np.pi*((ab2**2)-(mn/2)**2)/mn
    return {'AB2_m':ab2,'AB_m':2*ab2,'MN_m':mn,'K_m':k}

def apparent_resistivity(voltage_v,current_a,k_m):
    if np.any(np.asarray(current_a)<=0): raise ValueError('Corrente deve ser positiva')
    return np.asarray(k_m)*np.abs(np.asarray(voltage_v))/np.asarray(current_a)
