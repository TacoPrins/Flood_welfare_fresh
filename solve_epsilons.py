"""
solve.py

Purpose:
    Solve the model
"""
###########################################################
### Imports
import solve_model as solve_model
import collect_results

        
###########################################################
### main
def main():
    plot_distribution=True
    calculate_welfare=False    
    welfare_out = collect_results.collect_results(plot_distribution, calculate_welfare)
    #solve_model.solve()
###########################################################

### start main
if __name__ == "__main__":
    main()