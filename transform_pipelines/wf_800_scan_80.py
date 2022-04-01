def wf_800_scan_80(coords_input, *args, **kwargs):
        "Pre-calibrated third-order polynomial coordinate transform for etSTED"
        # example fit from "Transform coordinates" subwidget
        params_fit = [-4.867258043104539077e-09,
                        -2.285257544220520152e-09,
                        -2.111348786336117153e-09,
                        4.377495161563817806e-09,
                        7.959991018915018614e-06,
                        1.603133659683596637e-06,
                        -1.768328713767395065e-06,
                        9.867931075865758739e-02,
                        9.966619307046945507e-04,
                        -4.100423200817824210e+01,
                        -6.713418712002687816e-10,
                        2.661493461798385969e-09,
                        1.740168489639403944e-09,
                        -7.950415100680089105e-10,
                        1.941092459440164429e-07,
                        -2.333762648318877866e-06,
                        -1.138758151848054817e-06,
                        1.258471432774147403e-03,
                        -1.012687158228265938e-01,
                        4.297666571852118977e+01
                        ]

        c1 = coords_input[0]
        c2 = coords_input[1]
        x_i1 = params_fit[0]*c1**3 + params_fit[1]*c2**3 + params_fit[2]*c2*c1**2 + params_fit[3]*c1*c2**2 + params_fit[4]*c1**2 + params_fit[5]*c2**2 + params_fit[6]*c1*c2 + params_fit[7]*c1 + params_fit[8]*c2 + params_fit[9]
        x_i2 = params_fit[10]*c1**3 + params_fit[11]*c2**3 + params_fit[12]*c2*c1**2 + params_fit[13]*c1*c2**2 + params_fit[14]*c1**2 + params_fit[15]*c2**2 + params_fit[16]*c1*c2 + params_fit[17]*c1 + params_fit[18]*c2 + params_fit[19]
        return [x_i1, x_i2]