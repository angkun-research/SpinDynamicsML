using JLD2, CSV, DataFrames
using StaticArrays
using ProgressBars

nfolder = 250

# as: collections of 12 x 1000 x 6 arrays
# bs: collections of 12 x 999 x 6 arrays
# norm2s: collections of 12 x 6 arrays
# Bs: collection of scalars
# Flatten to 1 big ve
# Js : collection of vectors of 5 Js
results = DataFrame(Js=Vector{Float64}[], Xs = Vector{Float64}[])

data_count = 0
# Randomly sample Js from the Jsize dimensional space with each J in range -1 to 1
for folder_idx in ProgressBar(1:nfolder) # nfolder
    temp = load("data/HaoResults/results_task$(folder_idx)_of_$(nfolder).jld2")
    for sample_idx in 1:length(temp["Js_list"])
        Jtemp = temp["Js_list"][sample_idx] # vector of 5
        atemp = temp["as_list"][sample_idx] # array of 12 x 1000 x 6
        btemp = temp["bs_list"][sample_idx] # array of 12 x 999 x 6
        norm2temp = temp["norm2s_list"][sample_idx] # array of 12 x 6
        Btemp = temp["B_list"][sample_idx] # a scalar
        # combine atemp, btemp, norm2temp, Btemp into a single vector of features
        Xtemp = vcat(vec(atemp), vec(btemp), vec(norm2temp), [Btemp]) # this will be a vector of length 12*1000*6 + 12*999*6 + 12*6 + 1
        push!(results, (
            copy(Jtemp),
            copy(Xtemp)
        ))
        data_count += 1
        if data_count >= 1000
            break
        end
    end
    if data_count >= 1000
        break
    end
end
println("Data size", size(results))


CSV.write("data/HaoResults/CombinedData_J5_d1000.csv", results)
