/******************************************************************************/

/* N++ : C++ class neural network simulator                                   */
/* (c) 1994 Martin Riedmiller                                                 */
/* last changed: 27.7.94                                                      */

/* File: n++.h                                                                */
/* Purpose: n++ header file - to be included in application programs          */

/******************************************************************************/

#include<stdio.h>
#include<math.h>
#include<string.h>
#include<string>
#include<stdlib.h>

#ifndef NPP
#define NPP

#define ABSO(x) (x>0?x:-x)

/* EXTERNAL DECLARATIONS: */
#define FTYPE float       /* computing precision */
#define MAX_PARAMS 10     /* max. no of parameters for weight update,... */

/* Identification number for the functions (see include file "functions.c") */
/* Activation functions */
#define LOGISTIC 0
#define SYMMETRIC 1
#define LINEAR 2
/* Update functions */
#define BP 0
#define RPROP 1
#define RMSPROP 2

/* Error Codes */
#define OK 0
#define ERROR -1
#define FILE_ERROR -2

/* INTERNAL DECLARATIONS */
#define MAX_VARIABLES 3   /* number of user's variables in weight/unit structure */
#define MAX_LAYERS 100    /* max. no of layers */
#define BIAS 0       /* unit[0] is bias pseudo unit with output 1 */
/* Macros */
#define IS_INPUT(i) (i<=topo_data.in_count)
#define IS_UNIT(i) ((i>0)&&(i<=topo_data.unit_count))
#define IS_LAYER(l) ((l>=0)&&(l<topo_data.layer_count))

#ifdef MSDOS
/* fake drand48 using ansi lib and a macro */
#define drand48 (1.0/((double) RAND_MAX)) *rand
#define srand48(seed) srand((unsigned int) (seed))
#else
extern "C" double drand48( void ) __THROW;
extern "C" void srand48( long ) __THROW;
#endif

struct weight_typ{
  int from_unit;          /* no of incoming unit: -1 == bias unit */
  weight_typ *next; 
  FTYPE value;	
  FTYPE delta;
  FTYPE dEdw;
  FTYPE dEdw_sequence;    /* TD learning */
  FTYPE dOdw;             /* TD learning */
  FTYPE variable[MAX_VARIABLES]; /* for programmer's use */  
};

struct unit_typ{
  int unit_no;                   /* position of unit in unit array */
  weight_typ *weights;
  FTYPE dEdo;
  FTYPE dEdnet;                 
  FTYPE out;                     /* output_value */
  FTYPE net_in;                  /* summed input */
  FTYPE variable[MAX_VARIABLES]; /* programmer's use */
  FTYPE (*act_f)(unit_typ*);     /* pointer to activation function */
  FTYPE (*deriv_f)(unit_typ*);   /* pointer to derivation of activation function */
  unsigned short act_id;         /* identification number of activation function */
};			

struct layer_typ{
  unit_typ *unit;
  int unit_count;
};

struct topo_typ{
  int layer_count;  /* number of layers */
  int in_count,out_count,hidden_count,unit_count;   /* number of rsp. units */
};

struct scale_typ{
  int scale_function;
  int position;
  float parameter1;
  float parameter2;
  float parameter3;
  scale_typ *next;
};

class Net{
 protected:
  /* net data structure */
  unit_typ *unit;   /* array of all units in network */
  layer_typ layer[MAX_LAYERS];   /* virtual segmentation in layers */
  float update_params[MAX_PARAMS]; /* parameters for weight update function */
  scale_typ *scale_list_in, *scale_list_out;
  FTYPE *scaled_in_vec;
  void (*forward_pass_f)(unit_typ*,topo_typ*);  /* pointer to fw pass function */
  void (*backward_pass_f)(unit_typ*,topo_typ*); /* pointer to bw pass function */
  void (*update_f)(unit_typ*,topo_typ*,float*); /* pointer to update function */
  void create_units();
  void insert_scale_element(scale_typ **scale_list,int position,
                            int scale_function, float param1,float param2,float param3);
  void apply_scaling(float* data_vector, scale_typ *scale_list);
  void apply_backward_scaling(float* data_vector, scale_typ *scale_list);
  void delete_structure();
 public:
  topo_typ topo_data;
  int update_id;             /* identity of update function */
  FTYPE *in_vec, *out_vec;   /* communication vectors (have the appropriate size) */
  Net(void);
  ~Net(); 
  int create_layers(int layers, int layer_nodes[]);
  void connect_units(int to_unit, int from_unit, FTYPE value);
  void connect_layers();
  void connect_shortcut();
  void set_unit_act_f(int unit_no,int act_id);
  void set_layer_act_f(int layer_no,int act_id);
  int set_update_f(int typ,float *params);
  int save_net(std::string filename);
  int load_net( char filename[]);
  void print_net();
  void set_seed(long seed_no);
  void init_weights(int mode, FTYPE range);
  void forward_pass(FTYPE *in_vec, FTYPE *out_vec);
  void backward_pass(FTYPE *dedout, FTYPE *dedin);
  void backward_pass_light(FTYPE *dedout, FTYPE *dedin);
  void set_input_scaling(int position, int scale_function, 
                         float param1, float param2=0.0, float param3=0.0);
  void TD_backward_pass(FTYPE td_error, float lambda);
  void TD_init_sequence();
  void update_weights();
  void clear_derivatives();
  void multiply_derivatives(float factor);
  void TD_terminate_sequence(float weighting_factor);
};

#endif //NPP
