
run_mode = 'base'
def set_run_mode(mode = 'base'):
    global run_mode
    run_mode = mode
def get_run_mode():
    global run_mode
    return run_mode
    
hw = 'mant'
def set_hw_arch(hw = 'mant'):
    global hw_arch
    hw_arch = hw
def get_hw_arch():
    global hw_arch
    return hw_arch