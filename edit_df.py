import pandas as pd
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from typing import Self

class DataDescritpion():
    def __init__(
        self:Self,
        data_container:DeltaGenerator, 
        message_container:DeltaGenerator, 
        tab:DeltaGenerator
    ) -> None :
        self.data_container = data_container
        self.message_container = message_container
        self.tab = tab
        self.key = "data_editor_instance"
        self.display_dataeditor()
        self.add_buttons()
    
    def display_dataeditor(self:Self):
        with self.data_container:
            st.dataframe(data=st.session_state['editable_df'], height=250, use_container_width=True)

    #Data Frame functions
    def display_shape(self:Self):
        with self.message_container:
            st.text(st.session_state['editable_df'].shape)
            
    def display_description(self:Self):
        with self.message_container:
            st.dataframe(st.session_state['editable_df'].describe(), height=300)
            
    def display_info(self:Self):
        with self.message_container:
            df:pd.DataFrame = st.session_state['editable_df']
            info = {
                "Columns" : df.columns.to_list(),
                "Null Count": df.isnull().sum().to_list(),
                "Data Types": df.dtypes.to_list(),
                "Null Percent": (df.isnull().sum()/df.shape[0])*100
            }
            st.text("Columns Info")
            st.dataframe(data=info, height=230, use_container_width=True, hide_index=True)
            null_percent = (df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100 
            st.markdown(f"Null Value Percentage(Across Dataset): **{null_percent}**")
                
    def display_unique(self: Self):
        df: pd.DataFrame = st.session_state['editable_df']
        columns = list(df.columns)
        df = df.dropna()
        unique_count = {'Column':[], 'Unique Count': []}
        for column in columns:
            unique_count['Column'].append(column)
            unique_count['Unique Count'].append(len(df[column].unique().tolist()))
        self.message_container.dataframe(pd.DataFrame(unique_count), height=230, use_container_width=True, hide_index=True)
        self.message_container.info("**Note:** This count is only if all the null records dropped")
    
    def add_buttons(self:Self):
        s1, s2, s3, s4 = self.tab.columns(spec=4, vertical_alignment="center")
        
        s1.button(
            label="Info",
            help="Displays info `dataframe.info()`",
            on_click=self.display_info,
            use_container_width=True
        )
    
        s2.button(
            label="Shape",
            help="To display the shape of DataFrame `dataframe.shape`",
            on_click=self.display_shape,
            use_container_width=True
        )
        
        s3.button(
            label="Describe",
            help="Displays Description of DataFrame `dataframe.describe()`",
            on_click=self.display_description,
            use_container_width=True
        )
        
        s4.button(
            label="Unique",
            help="To get the count of unique values all the columns",
            on_click= self.display_unique,
            use_container_width=True
        )

class DataClean():
    def __init__(
        self:Self, 
        data_container:DeltaGenerator, 
        message_container:DeltaGenerator, 
        tab:DeltaGenerator
    ) -> None :
        self.data_container = data_container
        self.message_container = message_container
        self.tab = tab
        self.add_content()
        
    def drop_null(
        self:Self, 
        columns:list | None
    ) -> None:
        df:pd.DataFrame = st.session_state['editable_df']
        if df.isnull().sum().sum() == 0:
            with self.message_container:
                st.info("No Null Records")
            return
        try:
            if columns:
                df:pd.DataFrame = st.session_state['editable_df']
                df.dropna(inplace=True, subset=columns)
                with self.message_container:
                    st.success(f"Dropped Null Records from {columns}")
            else:
                st.session_state['editable_df'].dropna(inplace=True)
                with self.message_container:
                    st.success("Dropped Null Records")
        except Exception as e:
            with self.message_container:
                st.warning(f"Error occured: {e}")
    
    def fill_nulls(
        self:Self, 
        columns:list, 
        strategy:str ='Mean'
    ):
        from sklearn.impute import SimpleImputer
        
        df:pd.DataFrame = st.session_state['editable_df']
        if df.isnull().sum().sum() == 0:
            with self.message_container:
                st.info("No Null Records")
            return
        
        strategies = {"Mean": 'mean', "Median": 'median', "Mode": 'mode'}
        strategy = strategies[strategy]
        
        if len(columns)<=0:
            with self.message_container:
                st.warning("Please select atleast one column")
                return
        
        numerical = list(df.select_dtypes("float64")) + list(df.select_dtypes("int64"))
        num_cols = [column for column in columns if column in numerical]
        obj_cols = [column for column in columns if column not in numerical]
        
        print(num_cols)
        print(obj_cols)
        
        try:
            if len(num_cols) > 0:
                imputer = SimpleImputer(strategy = strategy)
                updated = pd.DataFrame(imputer.fit_transform(df[num_cols]), columns=num_cols)
                df.update(updated)
            if len(obj_cols) > 0:
                imputer = SimpleImputer(strategy="most_frequent")
                updated = pd.DataFrame(imputer.fit_transform(df[obj_cols]), columns=obj_cols)
                df.update(updated)
            st.session_state['editable_df'] = df
        except Exception as e:
            with self.message_container:
                st.error(f"Error occured: {e}")
                
    def delete(
        self:Self, 
        columns:list
    ):
        try:
            if columns:
                df:pd.DataFrame = st.session_state['editable_df']
                final_columns = [column for column in df.columns if column not in columns]
                st.session_state['editable_df'] = df[final_columns]
                with self.message_container:
                    st.success(f"Deleted {', '.join(columns)}")
            else:
                with self.message_container:
                    st.info("Please select atleast one column")
        except Exception as e:
            with self.message_container:
                st.error(f"Error occured: {e}")
            
    def display_dropna(
        self: Self, 
        space:DeltaGenerator
    ):
        columns = None
        columns = space.multiselect(
            label="Select Columns",
            options=list(st.session_state['editable_df'].columns),
            key="drop_columns_select",
            help="""Select the columns in which the nulls to be dropped.  
                    **Note**: _If not selected, all the nulls will be dropped_"""
        )
        
        space.button(
            label="Drop Null",
            help="Displays info `dataframe.dropna(inplace=True, subset=[col1, col2,...])`",
            on_click=self.drop_null,
            args=(columns, ),
            use_container_width=True,
            key="dropna"
        )
        
    def display_fillNaN(
        self:Self, 
        space:DeltaGenerator
    ):
        columns = None
        strategy = None
        columns = space.multiselect(
            label="Select Columns",
            options=list(st.session_state['editable_df'].columns),
            key="fill_columns_select"
        )
        strategy = space.radio(
            label="Select Strategy",
            help="""The strategy to fill the missing values. Default=Mean.  
                    Irrespective of your selection, 
                    the categorical columns will always be filled based on **Mode** of the column    
                """,
            options=["Mean", "Median", "Mode"],
            horizontal=True
        )
        
        space.button(
            label="Fill NaN",
            help="",
            on_click=self.fill_nulls,
            args=(columns, strategy),
            use_container_width=True
        )
        
    def display_delete(
        self:Self, 
        space:DeltaGenerator
    ):
        columns = None
        columns = space.multiselect(
            label="Select Columns",
            options=list(st.session_state['editable_df'].columns),
            key="delete_columns_select"
        )
        
        space.button(
            label="Delete",
            help="Deletes selected columns from DataFrame",
            on_click=self.delete,
            args=(columns, ),
            use_container_width=True,
            key="delete"
        )
        
    def add_content(self:Self):
        s1, s2, s3 = self.tab.tabs(["Drop Null", "Fill Nulls", "Delete"])
        self.display_dropna(s1)
        self.display_fillNaN(s2)
        self.display_delete(s3)
        

class DataProcess():
    def __init__(
        self:Self, 
        data_container:DeltaGenerator, 
        message_container:DeltaGenerator, 
        tab:DeltaGenerator
    ):
        self.data_container = data_container
        self.message_container: DeltaGenerator = message_container
        self.tab: DeltaGenerator = tab
        self.add_content()

    def encode(
        self:Self, 
        encoder: str, 
        column: str
    ):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        df:pd.DataFrame = st.session_state['editable_df']
        code = f"from sklearn.preprocessing import {''.join(encoder.split())}\n\n"
        try:
            if encoder == "Label Encoder":
                le = LabelEncoder()
                df[column+"_le"] = le.fit_transform(df[[column]])
                st.session_state['editable_df'] = df
                code += f"le = LabelEncoder()\ndf['{column}_le'] = le.fit_transform(df[['{column}']]) #df is DataFrame"
                self.message_container.success(f"Column '{column}' successfully _Label Encoded_. '{column+'_le'}' is added")
            elif encoder == "OneHot Encoder":
                oe = OneHotEncoder(drop='first')
                if len(df[column].unique()) != 2:
                    self.message_container.warning("More than 2 unique values in column. Use _Label encoder_ instead")
                    return
                df[column+"_oe"] = oe.fit_transform(df[[column]]).toarray()[:, :-1]
                st.session_state['editable_df'] = df
                code += f"oe = OneHotEncoder()\ndf['{column}_oe'] = oe.fit_transform(df[['{column}']])toarray()[:, :-1] #df is DataFrame"
                self.message_container.success(f"Column '{column}' successfully _OneHot Encoded_. '{column+'_le'}' is added")
            self.message_container.code(body=code, line_numbers=True)
        except Exception as e:
            self.message_container.error(f"Error during encoding: {e}")  
            
    def standardize(
        self:Self, 
        scaler:str, 
        column:str
    ):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        df:pd.DataFrame = st.session_state['editable_df']
        code = f"from sklearn.preprocessing import {''.join(scaler.split())}\n\n"
        try:
            if scaler == "Standard Scaler":
                ss = StandardScaler()
                df[column+"_ss"] = ss.fit_transform(df[[column]])
                code += f"ss = StandardScaler()\ndf['{column}_ss'] = ss.fit_transform(df[['{column}']]) #df is DataFrame"
                st.session_state['editable_df'] = df
                self.message_container.success(f"Column '{column}' successfully Standardized using _Standard Scaler_.  '{column+'_ss'}' is added")
            if scaler == "MinMax Scaler":
                mms = MinMaxScaler()
                df[column+"_mms"] = mms.fit_transform(df[[column]])
                code += f"mms = MinMaxScaler()\ndf['{column}_mms'] = mms.fit_transform(df[['{column}']]) #df is DataFrame"
                st.session_state['editable_df'] = df
                self.message_container.success(f"Column '{column}' successfully Standardized using _MinMax Scaler_.  '{column+'_mms'}' is added")
            self.message_container.code(body=code, line_numbers=True)     
        except Exception as e:
            self.message_container.error(f"Error during standardizing: {e}")
        
    def display_encode(
        self:Self, 
        s1: DeltaGenerator
    ):
        df:pd.DataFrame = st.session_state['editable_df']
        categorical_cols = list(df.select_dtypes(include="object").columns)
        
        if not categorical_cols:
            s1.warning("No categorical (object type) columns available for encoding.")      
            return      
        
        column = s1.selectbox(
            label="Select Column",
            options=categorical_cols,
            placeholder="Choose a column...",
            key="encode_column_select"
        )
        encoder = s1.radio(
            label="Scaler",
            options=["Label Encoder", "OneHot Encoder"],
            horizontal=True,
            key="encoder_select"
        )
        s1.button(
            label="Encode",
            help="Converts a categorical column to numerical",
            on_click=self.encode,
            args=(encoder, column,), 
            use_container_width=True,
            key="encode"
        )
    
    def display_standardize(
        self:Self, 
        s2:DeltaGenerator
    ) -> None:
        df:pd.DataFrame = st.session_state['editable_df']
        numerical_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
        
        if not numerical_cols:
            s2.warning("No numerical columns available for standardizing.")      
            return      
        
        column = s2.selectbox(
            label="Select Column",
            options=numerical_cols,
            placeholder="Choose a column...",
            help="""Select the column to be **standardized**.""",
        )
        scaler = s2.radio(
            label="Scaler",
            options=["Standard Scaler", "MinMax Scaler"],
            horizontal=True
        )
        s2.button(
            label="Standardize",
            help="Scales the numerical column without effecting the distribution",
            on_click=self.standardize,
            args=(scaler, column,), 
            use_container_width=True,
            key="standardize"
        )

    def add_content(self):
        options_container = self.tab.container()
        with options_container:
            s1, s2 = st.tabs(["Encode", "Standardize"])
        self.display_encode(s1)
        self.display_standardize(s2)
        
def revert(message_container:DeltaGenerator):
    try:
        st.session_state['editable_df'] = st.session_state['main_df']
        st.session_state['visual_df'] = st.session_state['main_df']
        with message_container:
            st.success("Reverted back all changes made")
    except Exception as e:
        with message_container:
            st.error(f"Error during reverting: {e}")
                
def update(message_container:DeltaGenerator):
    try:
        st.session_state['visual_df'] = st.session_state['editable_df']
        with message_container:
            st.success("Succesfully Update. You can go on with Visualization")
    except Exception as e:
        with message_container:
            st.error(f"Error during updating: {e}")

def editable_panel(tab: DeltaGenerator) -> None :

    data_container:DeltaGenerator = tab.container()
    message_container:DeltaGenerator = tab.container()
    content_container:DeltaGenerator = tab.container(border=True) 
    
    t1, t2, t3 = content_container.tabs(["Data Description", "Data Cleaning", "Data Preprocessing"])
    
    DataDescritpion(data_container, message_container, t1)
    DataClean(data_container, message_container, t2)
    DataProcess(data_container, message_container, t3)
    
    st.button(
        label="Revert",
        help="To revet back the DataFrame to its intitial state",
        on_click= revert,
        args = (message_container, ),
        use_container_width=True
    )
    
    st.button(
        label="Update internal DataFrame",
        help="To update the internal Data Frame with this edited Data Frame.  **Note:** _Only for Visualization_",
        on_click= update,
        args = (message_container, ),
        use_container_width=True
    )