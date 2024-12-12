
using Turing, StatsPlots, Random


# Set seed for reproducibility
Random.seed!(42)

# Simulated dataset: Customer interactions and churn
n = 100  # Number of customers
interactions = rand(1:10, n)  # Random interactions from 1 to 10
churn = [rand(Bernoulli(1 / (1 + exp(-0.5 * (x - 5))))) for x in interactions]

# Visualize the data
scatter(interactions, churn, xlabel="Number of Interactions", ylabel="Churn (0 or 1)", title="Customer Churn vs. Interactions")

# Define the Bayesian logistic regression model
@model function churn_model(interactions, churn)
    # Priors for regression parameters
    α ~ Normal(0, 5)  # Intercept
    β ~ Normal(0, 5)  # Slope
    
    # Model for observations
    for i in eachindex(churn)
        logit_p = α + β * interactions[i]
        churn[i] ~ Bernoulli(1 / (1 + exp(-logit_p)))  # Logistic regression
    end
end

# Set up and sample the model
model = churn_model(interactions, churn)
chain = sample(model, NUTS(), 1000)

# Summarize and visualize results
println(chain)
plot(chain)