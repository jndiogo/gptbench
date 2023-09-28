import copy

LOG=False

class Conf():
    """
    A dictionary with dict-like item access and attribute-style access, with optional type-enforcing and help for entries.

    Conf.name1, __init__, in, list(), items(), keys(), values(), dir(), get_type(), get_help(): accept/access local items, not from nested Confs

    Conf['name1.name2'], has(), get(), set(): accept path names like 'name1.name2.leaf_name' and local key names like 'name1'.

    Doesn't support copy.copy(): use Conf.copy()
    """
    
    def __init__(self, *args, **kwargs):
        self.__dict__['_registry'] = dict()
        self.__dict__['_dict'] = dict(*args, **kwargs)

    
    def setup(self, name, value, value_type, info=None):
        assert value is None or type(value) == value_type, "value can only be None or of value_type"

        self._registry[name] = (value_type, info)
        self._local_set(name, value)


    def has(self, path):
        """ Accepts path-like access: name1.name2.name3 """
        obj,leaf_name = self._obj_leaf_name_from_path(path)
        return leaf_name in obj
    

    def get(self, path, default_value=None):
        """ Accepts path-like access: name1.name2.name3 """
        try:
            return self._path_get(path)
        except KeyError:
            return default_value

    def set(self, path, value):
        """ Accepts path-like access: name1.name2.name3 """
        self._path_set(path,value)
    
            
    
    def update(self, *args, key_must_exist=True, filter_key_list=None, **kwargs):
        """
        Nested Conf items are only updated if they exist here.
        key_must_exist: only local existing keys are updated.
        filter_key_list: only update path/keys which are in this list 
        """

        if LOG: print(args, kwargs)
        
        for k,v in dict(*args, **kwargs).items():
            if LOG: print(k,v)

            if filter_key_list is not None:
                if k not in filter_key_list:
                    continue

            obj, leaf_name = self._obj_leaf_name_from_path(k)

            if key_must_exist:
                assert leaf_name in obj, f"Destination key path '{k}' does not exist"

            if LOG: print("obj,leaf_name", obj,leaf_name)

            if isinstance(v, type(self)):
                assert leaf_name in obj, f"Key path '{k}' of type Conf must exist to be updated"
                assert isinstance(obj[leaf_name], type(self)), "Atempting to update Conf into a different type"
                    
                obj[leaf_name].update(v.items(), key_must_exist=key_must_exist)
            else:
                obj._local_set(leaf_name,v)
            

    #def update_from_config(self, other_config, local_keys_list=None):
    #    """ local_keys_list: only update local keys which are in this list """
    #    self.update(other_config.items())

    
    def update_from_args(self, args_list, key_must_exist=True):
        """
        Update the config from a list of strings that is expected to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `-arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        -seed=1117 -work_dir=out -model.n_layer=10 -trainer.batch_size=32

        key_must_exist: only local existing keys are updated.
        """

        d = dict()
        for arg in args_list:

            keyval = arg.split('=')
            l = len(keyval)
            if l < 2: # -resume -> -resume=1
                keyval.append(1)
            elif l > 2: # take care of accepting multiple '=' like -start="1+1=?"
                keyval[1] = '='.join(keyval[1:])
                keyval = keyval[:2]

            key, val = keyval # unpack

            # find the appropriate object to insert the attribute into
            if key[:1] == '-': key = key[1:] # strip eventual first '-'
            if key[:1] == '-': key = key[1:] # strip eventual second '-'

            d[key]=val

        self.update(d, key_must_exist=key_must_exist)

    
    def to_dict(self, include_non_jsonable=True, filter_key_list=None):
        """ Return a dict representation of the Conf, including dicts of nested Confs """
        
        out = {}

        for k, v in self._dict.items():

            if filter_key_list is not None:
                if k not in filter_key_list:
                    continue

            if isinstance(v, type(self)):
                out[k] = v.to_dict(include_non_jsonable)
                
            elif isinstance(v, dict) or isinstance(v, list) or isinstance(v, tuple) or \
                 isinstance(v, str) or isinstance(v, float) or isinstance(v, int) or isinstance(v, bool) or \
                 v is None:
                out[k] = v
                
            elif include_non_jsonable: # include others that can't be converted to json
                out[k] = v

        return out

    
        
    def list(self):
        return self._dict.list()
    
    def items(self):
        return self._dict.items()
        
    def keys(self):
        return self._dict.keys()
        
    def values(self):
        return self._dict.values()

    def clear(self):
        self._dict.clear()




    def get_type(self, name):
        if name in self._registry:
            return self._registry[name][0]
        else:
            return None

    def get_help(self, name):
        if name in self._registry:
            return self._registry[name][1]
        else:
            return None


    #--------------------------------------------------------------------

    # dict access
    def __getitem__(self, path):
        """ Allows path access: conf['name1.name2'] """
        if LOG: print('__getitem__', path)
        return self._path_get(path)

    def __setitem__(self, path, value):
        """ Allows path access: conf['name1.name2'] """
        if LOG: print('__setitem__', path, value)
        self._path_set(path,value)

    def __delitem__(self, key):
         del self._dict[key]

    
    # object attribute access
    def __getattr__(self, key):
        if LOG: print('__getattr__', key)
        try:
            return self._dict[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if LOG: print('__setattr__', key, value)
        self._local_set(key,value)

    def __delattr__(self, key):
        try:
            del self._dict[key]
        except KeyError:
            raise AttributeError(key)

    

    def __dir__(self):
        if LOG: print('__dir__')
        return self._dict.keys()

    def __iter__(self):
        if LOG: print('__iter__')
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)


    def __copy__(self):
        if LOG: print('__copy__')
        return type(self)(**self._dict)

    def __deepcopy__(self, memo):
        if LOG: print('__deepcopy__')
        return type(self)(**copy.deepcopy(self._dict))


    def __str__(self):
        return self.dump(verbose=1)

    def dump(self, verbose, indent=0):
        """ 0: compact, 1: verbose, 2: verbose + _registry """
        
        indent_str = ' ' * indent

        own = []
        _own = []
        sub = []
        for k, v in self._dict.items():

            if k == '_registry' and verbose < 2:
                continue

            if isinstance(v, type(self)):

                label = "%s: " % k
                contents = v.dump(verbose, indent=0 if verbose < 1 else indent+4)

                if verbose < 1:
                    sub.append(indent_str + label + contents)
                else:
                    sub.append(label)
                    sub.append(indent_str + contents)

            else:
                if verbose:
                    st = "%s: %s (%s) " % (k, v, type(v).__name__)
                else:
                    st = "%s=%s " % (k, v)

                if k[0] != '_':
                    own.append(indent_str + st)
                else:
                    _own.append(indent_str + st)

        sep = '\n' if verbose else ''
        out = sep.join(own + _own)

        if len(sub):
            out += '\n'
            out += '\n'.join(sub)

        return out


    
    #--------------------------------------------------------------------
    
    def _obj_leaf_name_from_path(self, path):
        """ Returned obj is always a Conf. leaf_name always returned, even if doesn't exist """
        keys = path.split('.')
        obj = self
        for k in keys[:-1]:
            if not k in obj:
                return None, None
            obj = obj[k]
            assert type(obj) == type(self)

        return obj, keys[-1]

    
    def _local_set(self, name, value):
        """ Casts value if a type is found in registry """
        value_type = self.get_type(name)
        
        if value is not None and value_type is not None and type(value) != value_type:
            # type is set: cast value
            value = value_type(value)

        self._dict[name]=value
    
    def _path_set(self, path, value):
        obj,leaf_name = self._obj_leaf_name_from_path(path)
        if obj is not None:
            obj._dict[leaf_name] = value
        else:
            raise KeyError(path)

    def _path_get(self, path):
        obj,leaf_name = self._obj_leaf_name_from_path(path)
        if obj is not None and leaf_name in obj:
            return obj._dict[leaf_name]
        else:
            raise KeyError(path)

