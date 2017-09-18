"""
Gas models.
"""
EPM2_atoms = {
#FF_type  at_mass  sig[A]  eps[kcal/mol]  charge
        "Cx":(12.0000, 2.757, 0.0559, +0.6512),
        "Ox":(15.9994, 3.033, 0.16  , -0.3256)
        }

EPM2_angles = {
        # type K theta0
        "Ox_Cx_Ox": (295.41, 180)
        }
