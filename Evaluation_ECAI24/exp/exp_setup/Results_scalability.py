import numpy
import matplotlib.pyplot as plt

Drone_2_layer_ReLU = [5.500151 ,5.293492 ,5.383687 ,5.224187 ,5.262559 ,5.267393 ,5.279529 ,5.313675 ,5.172437 ,5.373752]
Drone_2_layer_Sigmoid = [5.496747 ,5.295281 ,5.29068 ,5.250134 ,5.244703 ,5.312446 ,5.247062 ,5.232222 ,5.286789 ,5.146375]
Drone_2_layer_Tanh = [5.380554 ,5.271393 ,5.204216 ,5.324204 ,5.31625 ,5.282917 ,5.280277 ,5.229607 ,5.362892 ,5.338464]
Drone_2_layer_leakyReLU = [5.434433 ,5.322476 ,5.253642 ,5.275892 ,5.256505 ,5.202247 ,5.338375 ,5.202326 ,5.135428 ,5.341394]
Drone_4_layer_ReLU = [8.013451 ,8.05693 ,7.995648 ,8.031459 ,8.206853 ,8.264775 ,8.191024 ,8.043848 ,8.238772 ,8.230006]
Drone_4_layer_Sigmoid = [7.954498 ,8.36245 ,8.082797 ,8.10523 ,8.112803 ,8.251647 ,8.155943 ,8.183585 ,8.041224 ,8.010218]
Drone_4_layer_Tanh = [7.932254 ,7.991961 ,8.201327 ,8.135298 ,8.170552 ,8.098813 ,8.059482 ,8.276054 ,8.37723 ,8.092547]
Drone_4_layer_leakyReLU = [8.061911 ,8.317932 ,8.214541 ,8.081088 ,8.258921 ,8.165959 ,8.27799 ,8.298839 ,8.392614 ,8.247391]
Drone_6_layer_ReLU = [10.610763 ,10.58039 ,10.465107 ,10.569402 ,10.374375 ,10.6229 ,10.647334 ,10.645201 ,10.614586 ,10.475736]
Drone_6_layer_Sigmoid = [10.460887 ,10.416908 ,10.619394 ,10.506693 ,10.603357 ,10.514558 ,10.588913 ,10.660628 ,10.458772 ,10.685219]
Drone_6_layer_Tanh = [10.471953 ,10.576653 ,10.695938 ,10.542854 ,10.64448 ,10.537425 ,10.452614 ,10.690263 ,10.659723 ,10.84944]
Drone_6_layer_leakyReLU = [10.551995 ,10.664763 ,10.407468 ,10.359652 ,10.484399 ,10.559807 ,10.57157 ,10.537932 ,10.478485 ,10.606874]
Drone_8_layer_ReLU = [13.329957 ,13.353651 ,13.22276 ,13.236297 ,13.19343 ,13.084147 ,13.116215 ,13.092672 ,13.313947 ,13.704323]
Drone_8_layer_Sigmoid = [13.327252 ,13.153761 ,13.246779 ,13.012555 ,13.159266 ,13.204209 ,12.919022 ,13.139127 ,13.076197 ,13.010886]
Drone_8_layer_Tanh = [13.127586 ,13.294021 ,13.202211 ,13.933132 ,13.14528 ,13.106434 ,13.126289 ,13.169539 ,12.989091 ,12.968458]
Drone_8_layer_leakyReLU = [12.992687 ,13.134617 ,13.102826 ,13.234094 ,13.099461 ,13.080652 ,13.156047 ,13.279573 ,13.141513 ,12.970332]
Drone_10_layer_ReLU = [15.702619 ,15.629268 ,15.749403 ,15.787956 ,15.577259 ,15.656822 ,15.756013 ,15.779149 ,15.946176 ,15.845809]
Drone_10_layer_Sigmoid = [15.692225 ,15.842584 ,15.590121 ,15.713763 ,15.724274 ,15.71811 ,15.695226 ,15.621078 ,15.533621 ,15.526832]
Drone_10_layer_Tanh = [15.841346 ,15.893593 ,15.663109 ,15.541372 ,15.732181 ,16.020015 ,15.590408 ,15.978353 ,15.641421 ,15.91806]
Drone_10_layer_leakyReLU = [15.624987 ,15.759696 ,15.662731 ,15.618002 ,15.578488 ,15.662899 ,15.673386 ,15.781001 ,15.526628 ,15.750963]
Drone_12_layer_ReLU = [18.365082 ,18.240622 ,17.941752 ,18.272845 ,18.666779 ,18.463766 ,18.008882 ,18.249745 ,18.117255 ,18.363113]
Drone_12_layer_Sigmoid = [18.110999 ,18.385682 ,18.024054 ,18.51136 ,18.130913 ,18.012294 ,17.918404 ,18.209019 ,18.32629 ,18.370567]
Drone_12_layer_Tanh = [18.442493 ,18.45389 ,18.375951 ,18.150182 ,18.37922 ,18.53993 ,18.353433 ,18.339184 ,18.17513 ,18.197852]
Drone_12_layer_leakyReLU = [18.408541 ,17.998606 ,18.057661 ,18.383121 ,18.263063 ,18.428415 ,18.323898 ,18.540263 ,18.337417 ,18.35595]

Drone_4_layer_14_neurons = [8.013451 ,8.05693 ,7.995648 ,8.031459 ,8.206853 ,8.264775 ,8.191024 ,8.043848 ,8.238772 ,8.230006]
Drone_4_layer_28_neurons = [8.077339 ,8.129917 ,8.347833 ,8.189552 ,8.275172 ,8.299933 ,8.220225 ,8.162405 ,8.220172 ,8.12532]
Drone_4_layer_56_neurons = [8.185401 ,8.639761 ,8.394098 ,8.191734 ,8.161479 ,8.365926 ,8.499773 ,8.23255 ,8.181873 ,8.374777]
Drone_4_layer_112_neurons = [8.879329 ,8.649061 ,8.829262 ,8.721057 ,8.759668 ,8.628526 ,8.690505 ,8.661782 ,8.651705 ,8.534992]
neurons_14 = numpy.mean(Drone_4_layer_14_neurons)
neurons_28 = numpy.mean(Drone_4_layer_28_neurons)
neurons_56 = numpy.mean(Drone_4_layer_56_neurons)
neurons_112 = numpy.mean(Drone_4_layer_112_neurons)
increase_56_to_112 = (neurons_112-neurons_56)/neurons_56
print(increase_56_to_112)

Drone_8_layer_4_hidden = [9.155648 ,9.108316 ,9.097092 ,9.005946 ,9.078629 ,9.157377 ,9.160659 ,9.062764 ,9.211145 ,9.161768]
Drone_8_layer_5_hidden = [8.984138 ,8.908323 ,9.012702 ,9.15829 ,9.216499 ,9.120269 ,9.588086 ,9.069733 ,9.054715 ,8.964984]
Drone_8_layer_6_hidden = [9.258056 ,9.018773 ,9.268111 ,8.995745 ,9.04368 ,8.897067 ,8.89825 ,8.943948 ,9.111576 ,8.895804]
Drone_8_layer_7_hidden = [9.319714 ,9.275267 ,9.256433 ,9.143689 ,9.115164 ,9.372698 ,9.064063 ,9.014691 ,9.179969 ,9.137446]
Drone_8_layer_8_hidden = [9.207052 ,8.946938 ,9.024803 ,9.060304 ,8.896508 ,8.99529 ,8.916654 ,8.967157 ,9.026697 ,8.938089]
Drone_8_layer_9_hidden = [9.010152 ,8.982247 ,9.121555 ,8.999784 ,8.989394 ,9.069147 ,9.116875 ,9.080115 ,9.348025 ,8.885074]
Drone_8_layer_10_hidden = [9.222593 ,9.28255 ,9.330298 ,9.261285 ,9.04182 ,9.216328 ,9.129583 ,8.996814 ,9.117759 ,9.377183]
Drone_8_layer_25_hidden = [9.207052 ,8.946938 ,9.024803 ,9.060304 ,8.896508 ,8.99529 ,8.916654 ,8.967157 ,9.026697 ,8.938089]
Drone_8_layer_50_hidden = [9.503799 ,9.518667 ,9.460482 ,9.418263 ,9.54999 ,9.932478 ,9.725117 ,9.47492 ,9.632828 ,9.560249]
Drone_8_layer_100_hidden = [10.146565 ,10.021642 ,10.064548 ,10.073049 ,10.167422 ,10.116065 ,9.939731 ,10.152478 ,10.158124 ,10.170781]

Drone_4_layer_0_NA = [0.075829 ,0.076013 ,0.075641 ,0.075589 ,0.076104 ,0.072911 ,0.074421 ,0.072704 ,0.074076 ,0.074484 ,0.073667]
Drone_4_layer_1_NA = [4.016133 ,3.982271 ,3.962634 ,3.993981 ,3.857015 ,3.82884 ,3.854043 ,3.906256 ,3.923125 ,3.911476]
Drone_4_layer_2_NA = [7.056414 ,6.953092 ,6.961679 ,7.120485 ,7.167477 ,7.163381 ,7.124684 ,7.118536 ,7.068281 ,7.103721]
Drone_4_layer_3_NA = [10.360138 ,10.505298 ,10.303732 ,10.325323 ,10.371877 ,10.233372 ,10.482156 ,10.330923 ,10.585993 ,10.452205]
Drone_4_layer_4_NA = [13.858079 ,13.867841 ,13.92423 ,13.838014 ,13.917431 ,13.998525 ,13.835974 ,13.998973 ,14.161335 ,13.717504]
Drone_4_layer_5_NA = [17.06702 ,17.12164 ,16.997654 ,16.69955 ,16.795761 ,16.99324 ,16.716869 ,16.886082 ,17.052672 ,16.894777]
Drone_4_layer_6_NA = [20.565087 ,20.026571 ,20.264437 ,19.767532 ,20.5656 ,20.30979 ,20.232163 ,19.76907 ,20.424899 ,21.160032]

Setup_n = [4, 5, 6, 7, 8, 9, 10, 25, 50, 100]
Runtime_n = [numpy.mean(Drone_8_layer_4_hidden), numpy.mean(Drone_8_layer_5_hidden),numpy.mean(Drone_8_layer_6_hidden),numpy.mean(Drone_8_layer_7_hidden),numpy.mean(Drone_8_layer_8_hidden),numpy.mean(Drone_8_layer_9_hidden),numpy.mean(Drone_8_layer_10_hidden),numpy.mean(Drone_8_layer_25_hidden),numpy.mean(Drone_8_layer_50_hidden),numpy.mean(Drone_8_layer_100_hidden)]

Setup_l = [2, 4, 6, 8, 10, 12]
ReLU = [numpy.mean(Drone_2_layer_ReLU), numpy.mean(Drone_4_layer_ReLU),numpy.mean(Drone_6_layer_ReLU),numpy.mean(Drone_8_layer_ReLU),numpy.mean(Drone_10_layer_ReLU),numpy.mean(Drone_12_layer_ReLU)]
Sigmoid = [numpy.mean(Drone_2_layer_Sigmoid), numpy.mean(Drone_4_layer_Sigmoid),numpy.mean(Drone_6_layer_ReLU),numpy.mean(Drone_8_layer_ReLU),numpy.mean(Drone_10_layer_ReLU),numpy.mean(Drone_12_layer_Sigmoid)]
Tanh = [numpy.mean(Drone_2_layer_Tanh), numpy.mean(Drone_4_layer_Tanh),numpy.mean(Drone_6_layer_Tanh),numpy.mean(Drone_8_layer_Tanh),numpy.mean(Drone_10_layer_Tanh),numpy.mean(Drone_12_layer_Tanh)]
leakyReLU = [numpy.mean(Drone_2_layer_leakyReLU), numpy.mean(Drone_4_layer_leakyReLU),numpy.mean(Drone_6_layer_leakyReLU),numpy.mean(Drone_8_layer_leakyReLU),numpy.mean(Drone_10_layer_leakyReLU),numpy.mean(Drone_12_layer_leakyReLU)]
Values = [ReLU, Sigmoid, Tanh]

Setup_a = [0, 1, 2, 3, 4, 5, 6]
Runtime_A = [numpy.mean(Drone_4_layer_0_NA), numpy.mean(Drone_4_layer_1_NA),numpy.mean(Drone_4_layer_2_NA),numpy.mean(Drone_4_layer_3_NA),numpy.mean(Drone_4_layer_4_NA),numpy.mean(Drone_4_layer_5_NA),numpy.mean(Drone_4_layer_6_NA)]
print(ReLU)
print(Sigmoid)
print(Tanh)

# Create plot
plt.figure(figsize=[6, 3.5])
# neurons / layers / neural_actions
subject = 'neurons'

if subject == 'neurons':
    plt.title('Runtime analysis with increasing number of hidden layers')
    plt.plot(Setup_l, ReLU, 'r--', label='ReLU activation functions', linewidth=1.5)  # Dashed red line
    plt.plot(Setup_l, Sigmoid, 'b--', label='Sigmoids activation functions', linewidth=1.5)  # Dashed blue line
    plt.plot(Setup_l, Tanh, 'g--', label='Tanh activation functions', linewidth=1.5)  # Dashed green line
    plt.plot(Setup_l, leakyReLU, 'y--', label='Leaky ReLU activation functions', linewidth=1.5)  # Dashed yellow line
    plt.xlabel('number of hidden layer')
    plt.ylabel('runtime (seconds)')
    plt.legend()  # Show legend
    plt.ylim(0, 20)
    plt.xlim(0, 13)
    # plt.yscale('log')
    plt.grid(True)  # Optional: Add grid for better readability
    plt.savefig("scale_l.svg", format = 'svg', dpi=1200)
elif subject == 'layers': 
    plt.title('Runtime analysis with increasing number of neurons per layer')
    plt.plot(Setup_n, Runtime_n, 'r--', linewidth=1.5)  # Dashed red line
    plt.xlabel('number of neurons per layer')
    plt.ylabel('runtime (seconds)')
    plt.xscale('log')
    plt.ylim(0, 12)
    plt.grid(True)  # Optional: Add grid for better readability
    plt.savefig("scale_n.svg", format = 'svg', dpi=1200)
elif subject == 'neural_actions':
    plt.title('Runtime analysis with increasing number of neural actions')
    plt.plot(Setup_a, Runtime_A, 'r--', linewidth=1.5)  # Dashed red line
    plt.xlabel('number of neural actions within the domain and the plan')
    plt.ylabel('runtime (seconds)')
    # plt.ylim(0, 6)
    plt.grid(True)  # Optional: Add grid for better readability
    plt.savefig("scale_a.svg", format = 'svg', dpi=1200)

plt.show()



# {
#     "ID": "test",
#     "DS_DOMAIN": "drone",
#     "Max_Step": 20,
#     "Int_ML": 1,
#     "N_SAMPLES": 1500,
#     "LOG_DIR": "exp/log",
#     "MODEL": "MLP",
#     "DEVICE": "cpu",
#     "SEED": 1,
#     "BATCH_SIZE": 256,
#     "LEARNING_RATE": 7.5e-5,
#     "WEIGHT_DECAY": 1e-12,
#     "EPOCHS": 4500,
#     "N_LAYERS": 2,
#     "N_HIDDEN": 14,
#     "DROPOUT": 0.1,
#     "N_MODELS": 16,
#     "REC_EPOCHS": 2000,
#     "REC_LR": 1.5e2,
#     "N_REC_VARS": 1,
#     "opti_threshold" : 5,
#     "SCALE" : 0,
#     "Listener": "full"
# }

# Drone_init_1 = ["0.0", "0.0", "10.0", "0.0", "0.0", "90"]
# Drone_goal_1 = ["0.0", "0.0", "0.0", "0.0", "0.0", ">90"]
# Drone_1 = df_problem(Drone_variables_names, Drone_init_1, Drone_goal_1)

# Drone_init_2 = ["0.0", "0.0", "10.0", "0.0", "0.0", "90"]
# Drone_goal_2 = ["0.0", "10.0", "0.0", "0.0", "0.0", ">90"]
# Drone_2 = df_problem(Drone_variables_names, Drone_init_2, Drone_goal_2)

# Drone_init_3 = ["0.0", "0.0", "0.0", "0.0", "0.0", "90"]
# Drone_goal_3 = ["10.0", "10.0", "0.0", "0.0", "0.0", "200"]
# Drone_3 = df_problem(Drone_variables_names, Drone_init_3, Drone_goal_3)

# Drone_init_4 = ["0.0", "0.0", "10.0", "10.0", "10.0", "50"]
# Drone_goal_4 = ["10.0", "10.0", "0.0", "0.0", "0.0", ">=100"]
# Drone_4 = df_problem(Drone_variables_names, Drone_init_4, Drone_goal_4)

# with Drone_var_parameters_names = ["energy"]
# Drone_var_parameters_top_compare = [1000.0]
# Drone_var_parameters_down_compare = [0.0]
# Drone_var_parameters_top_listener = [20.0]
# Drone_var_parameters_down_listener = [0.0]
# Drone_var_parameters = df_variable_parameters(Drone_var_parameters_names, Drone_var_parameters_top_compare, Drone_var_parameters_down_compare)