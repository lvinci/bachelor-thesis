/******************************************************************************/

/* N++ : C++ class neural network simulator                                   */
/* (c) 1994 Martin Riedmiller                                                 */
/* last changed: 22.06.95                                                      */

/* File: n++.cc                                                                */
/* Purpose: Implementation of n++ class neural network                         */

/******************************************************************************/

#include "npp.h"
#include <cstring>
#include <math.h>
//#include "functions.c"  /* activation, update, propagate,... */
/******************************************************************************/

/* N++ : C++ class neural network simulator                                   */
/* (c) 1994 Martin Riedmiller                                                 */
/* last changed: 27.7.94                                                      */

/* File: functions.c                                                          */
/* Purpose:                                                                    */
/* (User defined) activation functions, propagation functions,                 */
/* weight update functions,...                                                 */
/* included in net.cc                                                          */

/******************************************************************************/

/* macros: */
#define FORALL_WEIGHTS(WPTR)\
  int Counter;\
  for(Counter=topo->in_count+1;Counter<=topo->unit_count;Counter++)\
    for (WPTR=unit[Counter].weights; WPTR!=NULL; WPTR=WPTR->next)


/***************************************************************/
/*          WEIGHT UPDATE FUNCTIONS                            */
/***************************************************************/

void bp_init(float *params, unit_typ unit[], topo_typ *topo) {
    weight_typ *wptr;
    FORALL_WEIGHTS(wptr) {
            wptr->delta = wptr->dEdw = static_cast<FTYPE>(0);
        }
}

void bp_update(unit_typ unit[], topo_typ *topo, float *params)
/* update the weights according to gradient descent rule */
{
    weight_typ *wptr;
    const float learnrate = params[0];
    const float momentum = params[1];
    const float wd = params[2]; /* weight_decay */

    FORALL_WEIGHTS(wptr) {
            wptr->delta = -learnrate * wptr->dEdw + momentum * wptr->delta - wd * wptr->value;
            wptr->value += wptr->delta;
            wptr->dEdw = static_cast<FTYPE>(0); /* important: clear dEdw !*/
        }
}

#define UPDATE_VALUE 0.1
#define ETAPLUS 1.2
#define ETAMINUS 0.5
#define DELTA_MIN 1E-6
#define DELTA_MAX 50
#define EPSILON 0.001
#define BETA 0.9
#define PHI 1E-6
#define BATCH_SIZE 16
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

void rmsprop_init(float *params, unit_typ unit[], const topo_typ *topo) {
    weight_typ *wptr;

    if (!params[0])
        params[0] = EPSILON;
    if (!params[1])
        params[1] = BETA;
    if (!params[2])
        params[2] = PHI;
    if (!params[3])
        params[3] = BATCH_SIZE;
    FORALL_WEIGHTS(wptr) {
            wptr->delta = wptr->dEdw = static_cast<FTYPE>(0);
        }
}

void rmsprop_update(unit_typ unit[], topo_typ *topo, float *params) {
    weight_typ *wptr;
    const float learnrate = params[0];
    const float forget_factor = params[1];
    const float fuzz_factor = params[2];

    FORALL_WEIGHTS(wptr) {
            wptr->variable[0] = forget_factor * wptr->variable[0] + (1 - forget_factor) * wptr->dEdw * wptr->dEdw;
            wptr->delta = (learnrate / sqrt(wptr->variable[0] + fuzz_factor)) * wptr->dEdw;
            wptr->value -= wptr->delta;
            wptr->dEdw = static_cast<FTYPE>(0); /* important: clear dEdw !*/
        }
}

void rprop_init(float *params, unit_typ unit[], topo_typ *topo) {
    weight_typ *wptr;

    if (!params[0])
        params[0] = UPDATE_VALUE; /* default */
    if (!params[1])
        params[1] = DELTA_MAX; /* default */
    if (params[0] > params[1])
        params[0] = params[1];
    FORALL_WEIGHTS(wptr) {
            wptr->delta = wptr->dEdw = static_cast<FTYPE>(0);
            wptr->variable[0] = params[0];
        }
}

void rprop_update(unit_typ unit[], topo_typ *topo, float *params) {
    const float delta_max = params[1];
    const float wd = params[2];
    weight_typ *wptr;
    FORALL_WEIGHTS(wptr) {
            float update_value = wptr->variable[0];
            const float dEdw = wptr->dEdw + wd * wptr->value;
            const float direction = wptr->delta * dEdw;
            if (direction < 0.0) {
                update_value = MIN(update_value * ETAPLUS, delta_max);
                if (dEdw > 0.0)
                    wptr->delta = -update_value;
                else /* dEdw<0.0 */
                    wptr->delta = update_value;
            } else if (direction > 0.0) {
                update_value = MAX(update_value * ETAMINUS, DELTA_MIN);
                wptr->delta = 0.0; /* restart adaptation in next step */
            } else {
                /* direction == 0.0 */
                if (dEdw > 0.0)
                    wptr->delta = -update_value;
                else if (dEdw < 0.0)
                    wptr->delta = update_value;
                else
                    wptr->delta = static_cast<FTYPE>(0);
            }
            wptr->value += wptr->delta;
            wptr->variable[0] = update_value;
            wptr->dEdw = static_cast<FTYPE>(0); /* important: clear dEdw !*/
        }
}

/***************************************************************/
/*          ACTIVATION FUNCTIONS                               */
/***************************************************************/

FTYPE logistic(unit_typ *unit_i) {
    FTYPE net_in = unit_i->net_in;
    if (net_in >= 16.0) net_in = 16.0; /* avoid under/overflow */
    if (net_in <= -16.0) net_in = -16.0;
    return static_cast<FTYPE>((1.0 / (1.0 + exp(-net_in))));
}

FTYPE logistic_deriv(unit_typ *unit_i) {
    return (1.0 - unit_i->out) * unit_i->out;
}

FTYPE symmetric(unit_typ *unit_i) {
    FTYPE net_in = unit_i->net_in;
    if (net_in >= 7.0) net_in = 7.0; /* avoid under/overflow */
    if (net_in <= -7.0) net_in = -7.0;
    return static_cast<FTYPE>(-1.0 + 2.0 / (1.0 + exp(-net_in * 2.0)));
}

FTYPE symmetric_deriv(unit_typ *unit_i) {
    return 1.0 - (unit_i->out * unit_i->out);
}

FTYPE linear(unit_typ *unit_i) {
    return unit_i->net_in;
}

FTYPE linear_deriv(unit_typ *unit_i) {
    return 1.0;
}

FTYPE relu(unit_typ *unit_i) {
    const FTYPE net_in = unit_i->net_in;
    if (net_in < 0) return 0;
    return net_in;
}

FTYPE relu_deriv(unit_typ *unit_i) {
    const FTYPE net_out = unit_i->out;
    if (net_out < 0) return 0;
    return 1;
}

FTYPE LRELU_SLOPE = 0.01;

FTYPE l_relu(unit_typ *unit_i) {
    float sloap = LRELU_SLOPE;
    FTYPE net_in = unit_i->net_in;
    if (net_in < 0) return sloap * net_in;
    return net_in;
}

FTYPE l_relu_deriv(unit_typ *unit_i) {
    float sloap = LRELU_SLOPE;
    FTYPE net_out = unit_i->out;
    if (net_out < 0) return sloap;
    return 1;
}

/***************************************************************/
/*          PROPAGATION FUNCTIONS                              */
/***************************************************************/

void prop_forward(unit_typ unit[], topo_typ *topo) {
    for (int i = topo->in_count + 1; i <= topo->unit_count; i++) {
        unit[i].net_in = static_cast<FTYPE>(0); /* reset net in */
        for (const weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) /* compute new net in */
            unit[i].net_in += wptr->value * unit[wptr->from_unit].out;
        unit[i].out = unit[i].act_f(&unit[i]); /* compute unit activation and output */
    }
}

void prop_backward(unit_typ unit[], topo_typ *topo)
/* propagate backwards, compute dEdw and(!) dE/d(input) */
{
    for (int i = topo->unit_count; i > 0; i--) {
        unit[i].dEdnet = unit[i].dEdo * unit[i].deriv_f(&unit[i]); /* de/dnet_i */
        unit[i].dEdo = static_cast<FTYPE>(0); /* clear for next time */
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dEdw += unit[i].dEdnet * unit[wptr->from_unit].out; /* de/dwij */
            unit[wptr->from_unit].dEdo += unit[i].dEdnet * wptr->value; /* de/do_j */
        }
    }
}

void do_backward_pass_light(unit_typ unit[], const topo_typ *topo)
/* propagate backwards, compute ONLY!! dE/d(input) */
{
    for (int i = topo->unit_count; i > 0; i--) {
        unit[i].dEdnet = unit[i].dEdo * unit[i].deriv_f(&unit[i]); /* de/dnet_i */
        unit[i].dEdo = static_cast<FTYPE>(0); /* clear for next time */
        for (const weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            unit[wptr->from_unit].dEdo += unit[i].dEdnet * wptr->value; /* de/do_j */
        }
    }
}

/********************************************************************************/

/*       network creation functions                                             */

/********************************************************************************/

Net::Net() {
    // set defaults
    float default_params[MAX_PARAMS];

    topo_data.unit_count = topo_data.in_count =
    topo_data.out_count = topo_data.hidden_count = 0;
    topo_data.layer_count = 0;
    forward_pass_f = prop_forward;
    backward_pass_f = prop_backward;
    /* set default weight update function */
    for (float &default_param: default_params)
        default_param = 0.0;
    default_params[0] = 0.1;
    set_update_f(BP, default_params);
    scale_list_in = scale_list_out = nullptr;
    unit = nullptr;
    scaled_in_vec = in_vec = out_vec = nullptr;
}

Net::~Net() {
    delete_structure();
}

void Net::delete_structure() {
    if (in_vec != nullptr)
        delete[] in_vec;
    if (out_vec != nullptr)
        delete[] out_vec;
    if (scaled_in_vec != nullptr)
        delete[] scaled_in_vec;

    if (unit != nullptr) {
        for (int i = 0; i <= topo_data.unit_count; i++) {
            const weight_typ *wp1 = unit[i].weights;
            weight_typ *wp2;
            for (; wp1 != nullptr; wp1 = wp2) {
                wp2 = wp1->next;
                delete wp1;
            }
        }
        delete[] unit;
    }

    const scale_typ *scp1 = scale_list_in;
    scale_typ *scp2;
    for (; scp1 != nullptr; scp1 = scp2) {
        scp2 = scp1->next;
        delete scp1;
    }
    scp1 = scale_list_out;
    for (; scp1 != nullptr; scp1 = scp2) {
        scp2 = scp1->next;
        delete scp1;
    }

    /* ridi 11/96: reset */
    topo_data.unit_count = topo_data.in_count =
    topo_data.out_count = topo_data.hidden_count = 0;
    topo_data.layer_count = 0;
    scale_list_in = scale_list_out = nullptr;
    unit = nullptr;
    scaled_in_vec = in_vec = out_vec = nullptr;
    clear_derivatives(); // fix by TG
}

int Net::create_layers(int layers, int layer_nodes[]) {
    int i, start;

    if (topo_data.unit_count) {
        fprintf(stderr, "Can't create layers - network already defined\n");
        return (ERROR);
    }
    /* set topolgy variables */
    topo_data.layer_count = layers;
    topo_data.in_count = layer_nodes[0];
    for (i = 1, topo_data.hidden_count = 0; i < layers - 1; i++)
        topo_data.hidden_count += layer_nodes[i]; /* add number hidden units */
    topo_data.out_count = layer_nodes[layers - 1];
    /* allocate units */
    create_units(); /* create unit array */
    /* construct virtual segmentation in layers */
    for (i = 0, start = 1; i < topo_data.layer_count; i++) {
        layer[i].unit = &unit[start]; /* points to the first element in layer */
        layer[i].unit_count = layer_nodes[i];
        start += layer_nodes[i];
    }
    in_vec = new FTYPE[topo_data.in_count];
    scaled_in_vec = new FTYPE[topo_data.in_count];
    out_vec = new FTYPE[topo_data.out_count];
    return OK;
}

void Net::create_units() {
    topo_data.unit_count =
            topo_data.in_count + topo_data.hidden_count + topo_data.out_count; /* 'real' units */
    unit = new unit_typ[topo_data.unit_count + 1]; /* allocate unit array real + bias */
    /* create bias unit */
    unit[0].unit_no = 0;
    unit[0].out = static_cast<FTYPE>(1); /* bias output is constant 1 */
    unit[0].weights = nullptr; /* no weights  */

    for (int i = 1; i <= topo_data.unit_count; i++) {
        /* set defaults */
        unit[i].unit_no = i; /* position in unit array */
        unit[i].dEdo = unit[i].net_in = unit[i].out = unit[i].dEdnet = static_cast<FTYPE>(0);
        unit[i].weights = nullptr; /* no weights yet */
        for (float &k: unit[i].variable)
            k = static_cast<FTYPE>(0);
        if (IS_INPUT(i)) {
            unit[i].deriv_f = linear_deriv;
            unit[i].act_id = LINEAR;
        } else {
            unit[i].act_f = logistic;
            unit[i].deriv_f = logistic_deriv;
            unit[i].act_id = LOGISTIC;
        }
    }
}

void Net::connect_units(int to_unit, int from_unit, FTYPE value) {
    if (!IS_UNIT(to_unit) || (!IS_UNIT(from_unit) && from_unit != BIAS)) {
        fprintf(stderr, "connect_units: not possible from %d to %d\n", from_unit, to_unit);
        return;
    }
    weight_typ *temp = new weight_typ;
    temp->from_unit = from_unit;
    temp->value = value;
    temp->dEdw = static_cast<FTYPE>(0);
    temp->delta = static_cast<FTYPE>(0);
    temp->next = nullptr;
    for (float &i: temp->variable)
        i = static_cast<FTYPE>(0);
    if (unit[to_unit].weights == nullptr) {
        /* weight list empty */
        unit[to_unit].weights = temp;
    } else {
        weight_typ *wptr;
        /* append weight at end of weight list */
        for (wptr = unit[to_unit].weights; wptr->next != nullptr; wptr = wptr->next) {
            if (wptr->from_unit == from_unit) {
                fprintf(stderr, "Connection %d->%d already exists\n", from_unit, to_unit);
                /* destroy(temp) */
                return;
            }
        } /* end of list reached */
        if (wptr->from_unit == from_unit) {
            fprintf(stderr, "Connection %d->%d already exists\n", from_unit, to_unit);
            /* destroy(temp) */
        } else {
            wptr->next = temp;
        }
    }
}

void Net::connect_layers() {
    for (int l = 1; l < topo_data.layer_count; l++) /* start with first hidden layer */
        for (int i = 0; i < layer[l].unit_count; i++) {
            connect_units(layer[l].unit[i].unit_no, BIAS, 0);
            for (int j = 0; j < layer[l - 1].unit_count; j++)
                connect_units(layer[l].unit[i].unit_no, layer[l - 1].unit[j].unit_no, 0);
        }
}

void Net::connect_shortcut() {
    for (int l = 1; l < topo_data.layer_count; l++) /* start with first hidden layer */
        for (int i = 0; i < layer[l].unit_count; i++) {
            connect_units(layer[l].unit[i].unit_no, BIAS, 0);
            for (int k = 0; k < l; k++)
                for (int j = 0; j < layer[k].unit_count; j++)
                    connect_units(layer[l].unit[i].unit_no, layer[k].unit[j].unit_no, 0);
        }
}

/********************************************************************************/

/*       input/output scaling                                                   */

/********************************************************************************/

void Net::insert_scale_element(scale_typ **scale_list, int position,
                               int scale_function, float param1, float param2, float param3)
/* create a new scale element and insert at beginning of input/or output scale-list */
{
    scale_typ *new_elem = new scale_typ;
    new_elem->scale_function = scale_function;
    new_elem->parameter1 = param1;
    new_elem->parameter2 = param2;
    new_elem->parameter3 = param3;
    new_elem->position = position;
    new_elem->next = *scale_list;
    *scale_list = new_elem;
}

void Net::apply_scaling(float *data_vector, scale_typ *scale_list) {
    float new_value = 0;
    for (scale_typ *scale_elem = scale_list; scale_elem != nullptr; scale_elem = scale_elem->next) {
        switch (scale_elem->scale_function) {
            case 0: /* symmetric scaling : if x>0: y=a*x if x<0:y=b*x */
                new_value = (data_vector[scale_elem->position - 1] > 0
                             ? scale_elem->parameter1 * data_vector[scale_elem->position - 1]
                             : scale_elem->parameter2 * data_vector[scale_elem->position - 1]);
                break;
            case 1: /* linear scaling: y = a*x + b */
                new_value = scale_elem->parameter1 * data_vector[scale_elem->position - 1] +
                            scale_elem->parameter2;
                break;
            case 2: /* Binary scale if x<=param1 y=0; if x>= param2 y=1; else y = 0.5 */
                if (data_vector[scale_elem->position - 1] <= scale_elem->parameter1)
                    new_value = 0.0;
                else if (data_vector[scale_elem->position - 1] >= scale_elem->parameter2)
                    new_value = 1.0;
                else
                    new_value = 0.5;
                break;
            case 3: /* Binary symmetric scale if x<=param1 y=-1; if x>= param2 y=1; else y = 0 */
                if (data_vector[scale_elem->position - 1] <= scale_elem->parameter1)
                    new_value = -1.0;
                else if (data_vector[scale_elem->position - 1] >= scale_elem->parameter2)
                    new_value = 1.0;
                else
                    new_value = 0.;
                break;
            case 4: /* focus scaling : if x<a: y=s1*x if x>a:y=a*(s1-s2) + x*s2 */
                if (data_vector[scale_elem->position - 1] > scale_elem->parameter1)
                    new_value = scale_elem->parameter1 * (scale_elem->parameter2 - scale_elem->parameter3)
                                + scale_elem->parameter3 * data_vector[scale_elem->position - 1];
                else if (data_vector[scale_elem->position - 1] < -scale_elem->parameter1)
                    new_value = -scale_elem->parameter1 * (scale_elem->parameter2 - scale_elem->parameter3)
                                + scale_elem->parameter3 * data_vector[scale_elem->position - 1];
                else /* within focus */
                    new_value = scale_elem->parameter2 * data_vector[scale_elem->position - 1];
                break;
            default:
                break;
        } /* switch */
        data_vector[scale_elem->position - 1] = new_value;
    } /* for */
}

void Net::apply_backward_scaling(float *data_vector, scale_typ *scale_list) {
    /* compute dout'/dout */
    float new_value = 0;

    for (const scale_typ *scale_elem = scale_list; scale_elem != nullptr; scale_elem = scale_elem->next) {
        switch (scale_elem->scale_function) {
            case 0: /* symmetric scaling : if x>=0: y=a*x if x<0:y=b*x */
                new_value = (data_vector[scale_elem->position - 1] >= 0
                             ? scale_elem->parameter1 * data_vector[scale_elem->position - 1]
                             : scale_elem->parameter2 * data_vector[scale_elem->position - 1]);
                break;
            case 1: /* linear scaling: y = a*x + b */
                new_value = scale_elem->parameter1 * data_vector[scale_elem->position - 1];
                break;
            case 2:
                /* no gradient computation possible */
                new_value = data_vector[scale_elem->position - 1];
                break;
            case 3: /* Binary symmetric scale if x<=param1 y=-1; if x>= param2 y=1; else y = 0 */
                /* no gradient computation possible */
                new_value = data_vector[scale_elem->position - 1];
                break;
            default:
                break;
        } /* switch */
        data_vector[scale_elem->position - 1] = new_value;
    } /* for */
}

void Net::set_input_scaling(int position, int scale_function,
                            float param1, float param2, float param3) {
    insert_scale_element(&scale_list_in,
                         position,
                         scale_function,
                         param1,
                         param2,
                         param3);
}


/********************************************************************************/

/*       load/save network                                                      */

/********************************************************************************/

int Net::save_net(std::string filename) {
    scale_typ *scale_list;

    FILE *outfile = fopen(filename.c_str(), "w");
    if (outfile == nullptr) {
        fprintf(stderr, "Output file %s cannot be opened\n", filename.c_str());
        return (FILE_ERROR);
    }

    fprintf(outfile, "# Layers: %d\n", topo_data.layer_count);
    fprintf(outfile, "# Topology (input - hidden - output)\n");
    fprintf(outfile, "topology: ");
    for (int l = 0; l < topo_data.layer_count; l++)
        fprintf(outfile, "%d ", layer[l].unit_count);
    fprintf(outfile, "\nset_update_f %d ", update_id);
    for (float update_param: update_params)
        fprintf(outfile, "%f ", update_param);
    fprintf(outfile, "\n");
    for (scale_list = scale_list_in; scale_list != nullptr; scale_list = scale_list->next)
        fprintf(outfile, "input_scale %d %d %g %g %g\n", scale_list->position,
                scale_list->scale_function, scale_list->parameter1, scale_list->parameter2,
                scale_list->parameter3);
    fprintf(outfile, "\n");
    for (scale_list = scale_list_out; scale_list != nullptr; scale_list = scale_list->next)
        fprintf(outfile, "output_scale %d %d %g %g %g\n", scale_list->position,
                scale_list->scale_function, scale_list->parameter1, scale_list->parameter2,
                scale_list->parameter3);
    fprintf(outfile, "\n\n");
    fprintf(outfile, "# Connection structure - format:\n");
    fprintf(outfile, "# <unit_no> <act_id>\n");
    fprintf(outfile, "# Weights <from_unit> <value>\n\n");
    for (int l = 1; l < topo_data.layer_count; l++) /* start at first hidden layer */
        for (int i = 0; i < layer[l].unit_count; i++) {
            fprintf(outfile, "%d %d\n", layer[l].unit[i].unit_no, layer[l].unit[i].act_id);
            for (weight_typ *wptr = layer[l].unit[i].weights; wptr != nullptr; wptr = wptr->next)
                fprintf(outfile, "%d %g  ", wptr->from_unit, wptr->value);
            fprintf(outfile, "\n");
        }
    fclose(outfile);
    return (OK);
}

#define MAX_LINE_LEN 10000 /* max. length of line in network description */

int Net::load_net(const char filename[]) {
    char line[MAX_LINE_LEN];
    char *value;
    int read_in_topo[MAX_LAYERS];
    int i, current_unit, act_id;
    /* type of network description */
    int mode, no, from;
    FTYPE range;
    float p[MAX_PARAMS];
    int position, scale_function;
    float para1, para2, para3;
    char *saveptr;

    if (topo_data.unit_count) {
        /* ridi 11/96: overwrite old net */
        fprintf(stderr, "Net defined - OVERWRITING\n");
        delete_structure(); /* delete old net */
    }
    FILE *infile = fopen(filename, "r");
    if (infile == nullptr) {
        char secondchance[100];
        snprintf(secondchance, 100, "%s.net", filename);
        if ((infile = fopen(secondchance, "r")) == nullptr) {
            fprintf(stderr, "Neither %s nor %s can be opened\n", filename, secondchance);
            return (FILE_ERROR);
        }
    }

    int define_new = 0; /* type of description not yet identified */
    while (fgets(line, MAX_LINE_LEN, infile) != nullptr) {
        if (*line == '\n' || *line == '#') {
        } /* skip comments */
        else if (strncmp(line, "topology", 7) == 0) {
            strtok_r(line, " \t\n", &saveptr); /* skip first token (== topology)*/
            for (i = 0, value = strtok_r(nullptr, " \t", &saveptr); (value != nullptr) && (i < MAX_LAYERS);
                 value = strtok_r(nullptr, " \t\n", &saveptr), i++) {
                read_in_topo[i] = atoi(value);
            } /* finished reading topology  */
            if (create_layers(i, read_in_topo)) return (ERROR);
        } else if (strncmp(line, "set_update_f", 12) == 0) {
            strtok_r(line, " \t\n", &saveptr); /* skip first token (== set_update_f)*/
            value = strtok_r(nullptr, " \t\n", &saveptr); /* second token = type of update fun */
            mode = atoi(value);
            for (i = 0, value = strtok_r(nullptr, " \t", &saveptr); value != nullptr && (i < MAX_PARAMS);
                 value = strtok_r(nullptr, " \t\n", &saveptr), i++) {
                p[i] = static_cast<float>(atof(value));
            } /* finished reading update params  */
            set_update_f(mode, p);
        } else if (topo_data.unit_count) {
            /* units already defined */
            if (strncmp(line, "connect_layers", 12) == 0) {
                connect_layers();
                define_new = 1;
            } else if (strncmp(line, "connect_shortcut", 12) == 0) {
                connect_shortcut();
                define_new = 1;
            } else if (strncmp(line, "connect_units", 12) == 0) {
                sscanf(line, "%*s %d %d", &no, &from);
                connect_units(no, from, range);
            } else if (strncmp(line, "init_weights", 10) == 0) {
                sscanf(line, "%*s %d %f", &mode, &range);
                init_weights(mode, range);
            } else if (strncmp(line, "set_layer_act_f", 12) == 0) {
                sscanf(line, "%*s %d %d", &no, &mode);
                set_layer_act_f(no, mode);
            } else if (strncmp(line, "set_unit_act_f", 12) == 0) {
                sscanf(line, "%*s %d %d", &no, &mode);
                set_unit_act_f(no, mode);
            } else if (strncmp(line, "input_scale", 10) == 0) {
                sscanf(line, "%*s %d %d %f %f %f", &position, &scale_function, &para1, &para2, &para3);
                if ((position < 1) || (position > topo_data.in_count)) {
                    fprintf(stderr, "Error: Input_scale - undefined position %d\n", position);
                    return (ERROR);
                }
                insert_scale_element(&scale_list_in, position, scale_function, para1, para2, para3);
            } else if (strncmp(line, "output_scale", 10) == 0) {
                sscanf(line, "%*s %d %d %f %f %f", &position, &scale_function, &para1, &para2, &para3);
                if ((position < 1) || (position > topo_data.out_count)) {
                    fprintf(stderr, "Error: Output_scale - undefined position %d\n", position);
                    return (ERROR);
                }
                insert_scale_element(&scale_list_out, position, scale_function, para1, para2, para3);
            } else if (!define_new) {
                /* read  weights from file */
                /* read current unit_no and activation function identity */
                sscanf(line, "%d %d", &current_unit, &act_id);
                set_unit_act_f(current_unit, act_id);
                /* read weights (in next line) */
                if (fgets(line, MAX_LINE_LEN, infile)) {
                } // the if-statement is just to prevent a warning for an unused return value from fgets()
                for (value = strtok_r(line, " \t", &saveptr); value != nullptr && *value != '\n';
                     value = strtok_r(nullptr, " \t", &saveptr)) {
                    const int from_unit = atoi(value);
                    value = strtok_r(nullptr, " \t", &saveptr);
                    FTYPE weight_value = static_cast<FTYPE>(atof(value));
                    connect_units(current_unit, from_unit, weight_value);
                } /* for all weights */
            } /* if read weights from file */
        } /* if units already defined */
    } /* while read line from file */
    set_update_f(mode, p); // fix by TG
    fclose(infile); /* Martin, wie konntest du das vergessen ???? */
    return (OK);
}

void Net::print_net() {
    int i, l;

    printf("# Layers:\n%d\n", topo_data.layer_count);
    printf("# Topology (input - hidden - output)\n");
    for (l = 0; l < topo_data.layer_count; l++)
        printf("%d ", layer[l].unit_count);
    printf("\n");
    printf("Update Function: %d\nCurrent update Params: ", update_id);
    for (i = 0; i < MAX_PARAMS; i++)
        printf("%f  ", update_params[i]);
    printf("\n\n");
    printf("# Weights:\n");
    for (l = 0; l < topo_data.layer_count; l++)
        for (i = 0; i < layer[l].unit_count; i++) {
            printf("unit %d) act_id: %d   net_in:  %g   out %g\n",
                   layer[l].unit[i].unit_no, layer[l].unit[i].act_id,
                   layer[l].unit[i].net_in, layer[l].unit[i].out);
            for (const weight_typ *wptr = layer[l].unit[i].weights; wptr != nullptr; wptr = wptr->next)
                printf("%d %g  ", wptr->from_unit, wptr->value);
            printf("\n");
        }
}

/********************************************************************************/

/*       set/change network functions                                           */

/********************************************************************************/

void Net::set_unit_act_f(int unit_no, int act_id) {
    /* set activation function of unit unit_no */
    if (!IS_UNIT(unit_no)) {
        fprintf(stderr, "set act_f: Unit %d doesn't exists\n", unit_no);
        return;
    }
    if (IS_INPUT(unit_no)) {
        fprintf(stderr, "set act_f: No sense to set activation function of input unit %d\n", unit_no);
        return;
    }
    switch (act_id) {
        case LOGISTIC:
            unit[unit_no].act_f = logistic;
            unit[unit_no].deriv_f = logistic_deriv;
            unit[unit_no].act_id = LOGISTIC;
            break;
        case SYMMETRIC:
            unit[unit_no].act_f = symmetric;
            unit[unit_no].deriv_f = symmetric_deriv;
            unit[unit_no].act_id = SYMMETRIC;
            break;
        case LINEAR:
            unit[unit_no].act_f = linear;
            unit[unit_no].deriv_f = linear_deriv;
            unit[unit_no].act_id = LINEAR;
            break;
        case RELU:
            unit[unit_no].act_f = relu;
            unit[unit_no].deriv_f = relu_deriv;
            unit[unit_no].act_id = RELU;
            break;
        case LRELU:
            unit[unit_no].act_f = l_relu;
            unit[unit_no].deriv_f = l_relu_deriv;
            unit[unit_no].act_id = LRELU;
            break;
        default:
            fprintf(stderr, "set act_f: Unknown activation function %d\n", act_id);
    }
}

void Net::set_layer_act_f(int layer_no, int act_id) {
    /* set activation function of a complete layer */

    if (!IS_LAYER(layer_no)) {
        fprintf(stderr, "set_layer_act_f: Layer %d doesn't exist\n", layer_no);
        return;
    }
    for (int i = 0; i < layer[layer_no].unit_count; i++) {
        set_unit_act_f(layer[layer_no].unit[i].unit_no, act_id);
    }
}

int Net::set_update_f(int typ, float *params)
/* set current update function, init update_function, set update_params */
{
    switch (typ) {
        case BP:
            bp_init(params, unit, &topo_data); /* prepare bp, set update_params */
            update_f = bp_update;
            update_id = BP;
            break;
        case RPROP:
            rprop_init(params, unit, &topo_data); /* prepare rprop */
            update_f = rprop_update;
            update_id = RPROP;
            break;
        case RMSPROP:
            rmsprop_init(params, unit, &topo_data); /* prepare rmsprop */
            update_f = rmsprop_update;
            update_id = RMSPROP;
            break;
        default:
            fprintf(stderr, "set update_f: Unknown update function %d\n", typ);
            return (ERROR);
    }
    /* copy to internal update_params */
    for (int i = 0; i < MAX_PARAMS; i++)
        update_params[i] = params[i];
    return (0);
}

/********************************************************************************/

/*       working procedures                                                     */

/********************************************************************************/

void Net::set_seed(long seed_no) {
    srand48(seed_no);
}

void Net::init_weights(int mode, FTYPE range)
/* init the weights */
{
    int i;
    weight_typ *wptr;

    if (mode == 0) {
        for (i = topo_data.in_count + 1; i <= topo_data.unit_count; i++) /* for all except inputs */
            for (wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next)
                wptr->value = static_cast<FTYPE>((2.0 * range * drand48()) - range);
    } else if (mode == 1) {
        /* all biases = 0.0 */
        for (i = topo_data.in_count + 1; i <= topo_data.unit_count; i++) {
            /* for all except inputs */
            if ((wptr = unit[i].weights) != nullptr) {
                wptr->value = static_cast<FTYPE>(0);
                wptr = wptr->next;
                for (; wptr != nullptr; wptr = wptr->next)
                    wptr->value = static_cast<FTYPE>((2.0 * range * drand48()) - range);
            }
        }
    }
}

void Net::forward_pass(FTYPE *in_vec, FTYPE *out_vec)
/* take input vector, pass through network, return output vector */
{
    int i;

    if (scale_list_in != nullptr) {
        for (i = 0; i < topo_data.in_count; i++)
            scaled_in_vec[i] = in_vec[i];
        apply_scaling(scaled_in_vec, scale_list_in);
        for (i = 0; i < topo_data.in_count; i++)
            unit[i + 1].out = scaled_in_vec[i];
    } else {
        for (i = 0; i < topo_data.in_count; i++)
            unit[i + 1].out = in_vec[i];
    }
    forward_pass_f(unit, &topo_data); /* call (user specified) forward pass function */
    for (i = 0; i < topo_data.out_count; i++)
        out_vec[i] = unit[(topo_data.unit_count - topo_data.out_count + 1) + i].out;
    if (scale_list_out != nullptr)
        apply_scaling(out_vec, scale_list_out);
}

void Net::backward_pass(FTYPE *dedout, FTYPE *dedin)
/* take vector de/dout, backward pass, compute de/dw and de/dinput */
{
    int i;

    if (scale_list_out != nullptr)
        apply_backward_scaling(dedout, scale_list_out);
    for (i = 0; i < topo_data.out_count; i++)
        unit[(topo_data.unit_count - topo_data.out_count + 1) + i].dEdo = dedout[i];
    backward_pass_f(unit, &topo_data); /* call (user specified) backward pass function */
    for (i = 0; i < topo_data.in_count; i++)
        dedin[i] = unit[i + 1].dEdnet;
    if (scale_list_in != nullptr)
        apply_backward_scaling(dedin, scale_list_in);
}

void Net::backward_pass_light(FTYPE *dedout, FTYPE *dedin)
/* take vector de/dout, backward pass, compute de/dinput
SPECIAL: do not compute dEdw !!! (only dE/din) */
{
    int i;

    if (scale_list_out != nullptr)
        apply_backward_scaling(dedout, scale_list_out);
    for (i = 0; i < topo_data.out_count; i++)
        unit[(topo_data.unit_count - topo_data.out_count + 1) + i].dEdo = dedout[i];
    do_backward_pass_light(unit, &topo_data); /* call light backward pass function */
    for (i = 0; i < topo_data.in_count; i++)
        dedin[i] = unit[i + 1].dEdnet;
    if (scale_list_in != nullptr)
        apply_backward_scaling(dedin, scale_list_in);
}

void Net::update_weights()
/* call current update function to adapt the weights */
{
    update_f(unit, &topo_data, update_params);
}


/********************************************************************************/

/*       TD procedures                                                          */

/********************************************************************************/

void Net::TD_backward_pass(FTYPE td_error, float lambda)
/* computes dEdw_sequence
   according to TD learning rule
   different meanings:
   dEdo == dO/do_i;       (influence of output i on network output O)
   dEdnet == dO/dnet_i;   (...)
   dEdw_sequence = summed error of sequence; must be stored by explicit call
                   of TD_terminate_sequence(weighting factor)
*/
{
    unit[topo_data.unit_count].dEdo = static_cast<FTYPE>(1); /* for output: dO/do_i = 1 */
    for (int i = topo_data.unit_count; i > topo_data.in_count; i--) {
        unit[i].dEdnet = unit[i].dEdo * unit[i].deriv_f(&unit[i]); /* dO/dnet_i */
        unit[i].dEdo = static_cast<FTYPE>(0); /* clear for next time */
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dEdw_sequence += wptr->dOdw * td_error;
            wptr->dOdw = lambda * wptr->dOdw + unit[i].dEdnet * unit[wptr->from_unit].out;
            unit[wptr->from_unit].dEdo += unit[i].dEdnet * wptr->value; /* de/do_j */
        }
    }
}

void Net::TD_init_sequence() {
    for (int i = topo_data.unit_count; i > 0; i--) {
        unit[i].dEdnet = unit[i].dEdo = static_cast<FTYPE>(0); /* clear for next time */
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dOdw = static_cast<FTYPE>(0); /* clear for next time */
            wptr->dEdw_sequence = static_cast<float>(0); /* clear for next time */
        }
    }
}

void Net::TD_terminate_sequence(float weighting_factor)
/* add sequence to dEdw, weighted by a weighting_factor */
{
    for (int i = topo_data.unit_count; i > 0; i--) {
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dEdw += weighting_factor * wptr->dEdw_sequence;
        }
    }
}

void Net::clear_derivatives()
/* special: set all derivatives to zero;
   used e.g. for useless sequences in TD-learning */
{
    for (int i = topo_data.unit_count; i > 0; i--) {
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dEdw = static_cast<FTYPE>(0); /* clear for next time */
        }
    }
}

void Net::multiply_derivatives(float factor)
/* special: multiply all derivatives with factor;
   used e.g. to stress sequences in TD-learning */
{
    for (int i = topo_data.unit_count; i > 0; i--) {
        for (weight_typ *wptr = unit[i].weights; wptr != nullptr; wptr = wptr->next) {
            wptr->dEdw *= factor; /* clear for next time */
        }
    }
}
