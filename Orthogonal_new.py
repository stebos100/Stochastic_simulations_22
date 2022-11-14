def generate_ortho(self, num_samples):
    
    real_min = self.minx
    real_max = self.maxx
    imaginary_min = self.miny
    imaginary_max = self.maxy
        
    # make x an y spaces to form the 2d hypercube just like LHS
    x = np.linspace(real_min, real_max, nsamples+1)
    y = np.linspace((imaginary_min), (imaginary_max), nsamples+1)

    # make a list with x ranges consisting of pairs of x values that are consecutive in the linspace. 
    x_ranges = []
    y_ranges = []
        
    for i in range(1, len(x)):
        x_ranges.append([x[i - 1], x[i]])
        y_ranges.append([y[i - 1], y[i]])
    # make the subspaces in the hypercube, this is specific for orthogonal
    x_subspaces = np.arange(0, nsamples+np.sqrt(nsamples), np.sqrt(nsamples))
    y_subspaces = np.arange(0, nsamples+np.sqrt(nsamples), np.sqrt(nsamples))
    
    # Make the indices that will be shuffled each run
    x_indices = [i for i in range(nsamples)]
    y_indices = [i for i in range(nsamples)]
    
    #Create a list with the intervals between the blocks 
    block_interval = []
    for i in range(1, len(x_subspaces)):
        block_interval.append([int(x_subspaces[i-1]), int(x_subspaces[i])])
        
    random.shuffle(x_indices)
    random.shuffle(y_indices)
    
    samples = []
    
    for i in block_interval:
        for j in block_interval:
            # create list with individual blocks in the interval range
            x_blocks = [x for x in range(j[0], j[1])]
            y_blocks = [y for y in range(i[0], i[1])]

                    
            coordinate_x = 0
            coordinate_y = 0

            # loop through available indices
            for k in x_indices:
            # if an available index is in slice of indices
            #print(k)
                if k in x_blocks:
                    # set x coordinate to available index
                    coordinate_x = k
                    # remove available index
                    x_indices.remove(k)
                    # break out of the for-loop
                    break

            # similar for y
            for k in y_indices:
                if k in y_blocks:
                    coordinate_y = k
                    y_indices.remove(k)
                    break

            x_sample_range = x_ranges[coordinate_x]
            y_sample_range = y_ranges[-(coordinate_y+1)]

            sample = complex(random.uniform(x_sample_range[0], x_sample_range[1]), random.uniform(y_sample_range[0], y_sample_range[1]))
            samples.append(sample)
    
    return samples
    
def compute_area_ortho(self, num_runs, nsamples, numiterations):
        
    real_min = self.minx
    real_max = self.maxx
    imaginary_min = self.miny
    imaginary_max = self.maxy
        
    areas = []

    for i in range(num_runs):
        in_mandel = 0
        total_drawn = nsamples
        area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)

        samples = self.generate_ortho(nsamples)
            
        for c in samples:
            if self.within_mandel(numiterations, sample):
                in_mandel += 1
                    
        ratio_inmandel = (in_mandel/total_drawn)
        area_mandel = ratio_inmandel*area_T        

        areas.append(area_mandel)

    return areas