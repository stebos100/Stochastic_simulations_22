def LHS(self, num_runs, nsamples, numiterations):

    real_min = self.minx
    real_max = self.maxx
    imaginary_min = self.miny
    imaginary_max = self.maxy

    # make x an y spaces to form the hypercube 
    x = np.linspace(real_min, real_max, nsamples+1)
    y = np.linspace((imaginary_min), (imaginary_max), nsamples+1)


    # make a list with x ranges consisting of pairs of x values that are consecutive in the linspace. 
    x_ranges = []
    y_ranges = []
    for i in range(1, len(x)):
        x_ranges.append([x[i - 1], x[i]])
        y_ranges.append([y[i - 1], y[i]])

    # Randomly shuffle the x and y pairs
    random.shuffle(x_ranges)
    random.shuffle(y_ranges)
    
    areas=[]

    for i in range(num_runs):
        in_mandel = 0
        total_drawn = nsamples
        area_T =  np.abs((real_min - real_max))*np.abs(imaginary_max - imaginary_min)
        samp = []

        for j in range(nsamples):

            # take last strata in list
            x_range = x_ranges.pop()
            y_range = y_ranges.pop()
            
            sample = complex(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])) 

            # check if in mandelbrotset
            if self.within_mandel(numiterations, sample):
                in_mandel +=1

            samp.append(sample)

        ratio_inmandel = (in_mandel/total_drawn)
        area_mandel = ratio_inmandel*area_T        

        areas.append(area_mandel)
    return sample_area