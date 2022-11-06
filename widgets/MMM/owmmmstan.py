import sys, json
import threading
import itertools
import concurrent.futures

from collections import namedtuple
from typing import List, Optional

from math import isnan

import numpy
from scipy.sparse import issparse
from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from AnyQt.QtWidgets import (
    QTableView, QHeaderView, QAbstractButton, QApplication, QStyleOptionHeader,
    QStyle, QStylePainter
)
from AnyQt.QtGui import QColor, QClipboard
from AnyQt.QtCore import (
    Qt, QSize, QEvent, QObject, QMetaObject,
    QAbstractProxyModel, QIdentityProxyModel, QModelIndex,
    QItemSelectionModel, QItemSelection, QItemSelectionRange,
)
from AnyQt.QtCore import pyqtSlot as Slot

import Orange.data
from Orange.data.storage import Storage
from Orange.data.table import Table
from Orange.data.sql.table import SqlTable
from Orange.statistics import basic_stats
# kh
# from Orange.data.data_review import DataReviewer
from Orange.widgets.datavalidation.data_review import DataReviewer


from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemdelegates import TableDataDelegate
from Orange.widgets.utils.itemselectionmodel import (
    BlockSelectionModel, ranges, selection_blocks
)
from Orange.widgets.utils.tableview import TableView, \
    table_selection_to_mime_data
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, MultiInput, Output
from Orange.widgets.utils import datacaching
from Orange.widgets.utils.annotated_data import (create_annotated_table,
                                                 ANNOTATED_DATA_SIGNAL_NAME)
from Orange.widgets.utils.itemmodels import TableModel
from Orange.widgets.utils.state_summary import format_summary_details

# kh
from Orange.widgets.data import owfile

class RichTableModel(TableModel):
    """A TableModel with some extra bells and whistles/

    (adds support for gui.BarRole, include variable labels and icons
    in the header)
    """
    #: Rich header data flags.
    Name, Labels, Icon = 1, 2, 4

    def __init__(self, sourcedata, parent=None):
        super().__init__(sourcedata, parent)

        self._header_flags = RichTableModel.Name
        self._continuous = [var.is_continuous for var in self.vars]
        labels = []
        for var in self.vars:
            if isinstance(var, Orange.data.Variable):
                labels.extend(var.attributes.keys())
        self._labels = list(sorted(
            {label for label in labels if not label.startswith("_")}))

    def data(self, index, role=Qt.DisplayRole,
             # for faster local lookup
             _BarRole=gui.TableBarItem.BarRole):
        # pylint: disable=arguments-differ
        if role == _BarRole and self._continuous[index.column()]:
            val = super().data(index, TableModel.ValueRole)
            if val is None or isnan(val):
                return None

            dist = super().data(index, TableModel.VariableStatsRole)
            if dist is not None and dist.max > dist.min:
                return (val - dist.min) / (dist.max - dist.min)
            else:
                return None
        elif role == Qt.TextAlignmentRole and self._continuous[index.column()]:
            return Qt.AlignRight | Qt.AlignVCenter
        else:
            return super().data(index, role)

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is None:
                return super().headerData(
                    section, orientation, Qt.DisplayRole)

            lines = []
            if self._header_flags & RichTableModel.Name:
                lines.append(var.name)
            if self._header_flags & RichTableModel.Labels:
                lines.extend(str(var.attributes.get(label, ""))
                             for label in self._labels)
            return "\n".join(lines)
        elif orientation == Qt.Horizontal and role == Qt.DecorationRole and \
                self._header_flags & RichTableModel.Icon:
            var = super().headerData(
                section, orientation, TableModel.VariableRole)
            if var is not None:
                return gui.attributeIconDict[var]
            else:
                return None
        else:
            return super().headerData(section, orientation, role)

    def setRichHeaderFlags(self, flags):
        if flags != self._header_flags:
            self._header_flags = flags
            self.headerDataChanged.emit(
                Qt.Horizontal, 0, self.columnCount() - 1)

    def richHeaderFlags(self):
        return self._header_flags


class TableSliceProxy(QIdentityProxyModel):
    def __init__(self, parent=None, rowSlice=slice(0, -1), **kwargs):
        super().__init__(parent, **kwargs)
        self.__rowslice = rowSlice

    def setRowSlice(self, rowslice):
        if rowslice.step is not None and rowslice.step != 1:
            raise ValueError("invalid stride")

        if self.__rowslice != rowslice:
            self.beginResetModel()
            self.__rowslice = rowslice
            self.endResetModel()

    def mapToSource(self, proxyindex):
        model = self.sourceModel()
        if model is None or not proxyindex.isValid():
            return QModelIndex()

        row, col = proxyindex.row(), proxyindex.column()
        row = row + self.__rowslice.start
        assert 0 <= row < model.rowCount()
        return model.createIndex(row, col, proxyindex.internalPointer())

    def mapFromSource(self, sourceindex):
        model = self.sourceModel()
        if model is None or not sourceindex.isValid():
            return QModelIndex()
        row, col = sourceindex.row(), sourceindex.column()
        row = row - self.__rowslice.start
        assert 0 <= row < self.rowCount()
        return self.createIndex(row, col, sourceindex.internalPointer())

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        count = super().rowCount()
        start, stop, step = self.__rowslice.indices(count)
        assert step == 1
        return stop - start


TableSlot = namedtuple("TableSlot", ["input_id", "table", "summary", "view"])


class DataTableView(gui.HScrollStepMixin, TableView):
    dataset: Table
    input_slot: TableSlot


class TableBarItemDelegate(gui.TableBarItem, TableDataDelegate):
    pass


class OWDataTable(OWWidget):
    name = "Google Lightweight MMM"
    description = "View the dataset in a spreadsheet."
    icon = "icons/LogisticRegression.svg"
    priority = 50
    keywords = []

    class Inputs:
        data = MultiInput("Data", Table, auto_summary=False, filter_none=True)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)

    buttons_area_orientation = Qt.Vertical

    show_distributions = Setting(False)
    dist_color_RGB = Setting((220, 220, 220, 255))
    show_attribute_labels = Setting(True)
    select_rows = Setting(True)
    auto_commit = Setting(True)

    color_by_class = Setting(True)
    selected_rows = Setting([], schema_only=True)
    selected_cols = Setting([], schema_only=True)

    settings_version = 2

    def __init__(self):
        super().__init__()
        self._inputs: List[TableSlot] = []
        self.__pending_selected_rows = self.selected_rows
        self.selected_rows = None
        self.__pending_selected_cols = self.selected_cols
        self.selected_cols = None

        self.dist_color = QColor(*self.dist_color_RGB)

        info_box = gui.vBox(self.controlArea, "Missing data")
        self.info_text = gui.widgetLabel(info_box)
        self._set_input_summary(None)


        # box = gui.vBox(self.controlArea, "Checking Type")
        # self.c_show_attribute_labels = gui.checkBox(
        #     box, self, "show_attribute_labels",
        #     "Show variable labels (if present)",
        #     callback=self._on_show_variable_labels_changed)

        # self.c_show_attribute_labels = gui.checkBox(box, self, "show_distributions",
        #              'Visualize numeric values',
        #              callback=self._on_distribution_color_changed)
        # gui.checkBox(box, self, "color_by_class", 'Color by instance classes',
        #              callback=self._on_distribution_color_changed)

        # box = gui.vBox(self.controlArea, "Selection")

        # gui.checkBox(box, self, "select_rows", "Select full rows",
        #              callback=self._on_select_rows_changed)

        gui.rubber(self.controlArea)

        gui.button(self.buttonsArea, self, "MMM Stan",
                    callback=self._mmm_stan_model)

        # gui.button(self.buttonsArea, self, "Restore Original Order",
        #            callback=self.restore_order,
        #            tooltip="Show rows in the original order",
        #            autoDefault=False,
        #            attribute=Qt.WA_LayoutUsesWidgetRect)
        # gui.auto_send(self.buttonsArea, self, "auto_commit")

        # GUI with tabs
        self.tabs = gui.tabWidget(self.mainArea)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

    def copy_to_clipboard(self):
        self.copy()

    @staticmethod
    def sizeHint():
        return QSize(800, 500)

    def _create_table_view(self):
        view = DataTableView()
        view.setSortingEnabled(True)
        view.setItemDelegate(TableDataDelegate(view))

        if self.select_rows:
            view.setSelectionBehavior(QTableView.SelectRows)

        header = view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.AscendingOrder)

        # QHeaderView does not 'reset' the model sort column,
        # because there is no guaranty (requirement) that the
        # models understand the -1 sort column.
        def sort_reset(index, order):
            if view.model() is not None and index == -1:
                view.model().sort(index, order)
        header.sortIndicatorChanged.connect(sort_reset)
        return view

    @Inputs.data
    def set_dataset(self, index: int, data: Table):
        """Set the input dataset."""
        datasetname = getattr(data, "name", "Data")
        slot = self._inputs[index]
        view = slot.view
        # reset the (header) view state.
        view.setModel(None)
        view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        assert self.tabs.indexOf(view) != -1
        self.tabs.setTabText(self.tabs.indexOf(view), datasetname)
        view.dataset = data
        slot = TableSlot(index, data, table_summary(data), view)
        view.input_slot = slot
        self._inputs[index] = slot
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)

    @Inputs.data.insert
    def insert_dataset(self, index: int, data: Table):
        datasetname = getattr(data, "name", "Data")
        view = self._create_table_view()
        slot = TableSlot(None, data, table_summary(data), view)
        view.dataset = data
        view.input_slot = slot
        self._inputs.insert(index, slot)
        self.tabs.insertTab(index, view, datasetname)
        self._setup_table_view(view, data)
        self.tabs.setCurrentWidget(view)

    @Inputs.data.remove
    def remove_dataset(self, index):
        slot = self._inputs.pop(index)
        view = slot.view
        self.tabs.removeTab(self.tabs.indexOf(view))
        view.setModel(None)
        view.hide()
        view.deleteLater()

        current = self.tabs.currentWidget()
        if current is not None:
            self._set_input_summary(current.input_slot)

    def handleNewSignals(self):
        super().handleNewSignals()
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)
        data: Optional[Table] = None
        current = self.tabs.currentWidget()
        slot = None
        if current is not None:
            data = current.dataset
            slot = current.input_slot

        if slot and isinstance(slot.summary.len, concurrent.futures.Future):
            def update(_):
                QMetaObject.invokeMethod(
                    self, "_update_info", Qt.QueuedConnection)
            slot.summary.len.add_done_callback(update)
        self._set_input_summary(slot)

        if data is not None and self.__pending_selected_rows is not None:
            self.selected_rows = self.__pending_selected_rows
            self.__pending_selected_rows = None
        else:
            self.selected_rows = []

        if data and self.__pending_selected_cols is not None:
            self.selected_cols = self.__pending_selected_cols
            self.__pending_selected_cols = None
        else:
            self.selected_cols = []

        self.set_selection()
        self.commit.now()

    def _setup_table_view(self, view, data):
        """Setup the `view` (QTableView) with `data` (Orange.data.Table)
        """
        datamodel = RichTableModel(data)
        rowcount = data.approx_len()

        if self.color_by_class and data.domain.has_discrete_class:
            color_schema = [
                QColor(*c) for c in data.domain.class_var.colors]
        else:
            color_schema = None
        if self.show_distributions:
            view.setItemDelegate(
                TableBarItemDelegate(
                    view, color=self.dist_color, color_schema=color_schema)
            )
        else:
            view.setItemDelegate(TableDataDelegate(view))

        # Enable/disable view sorting based on data's type
        view.setSortingEnabled(is_sortable(data))
        header = view.horizontalHeader()
        header.setSectionsClickable(is_sortable(data))
        header.setSortIndicatorShown(is_sortable(data))
        header.sortIndicatorChanged.connect(self.update_selection)

        view.setModel(datamodel)

        vheader = view.verticalHeader()
        option = view.viewOptions()
        size = view.style().sizeFromContents(
            QStyle.CT_ItemViewItem, option,
            QSize(20, 20), view)

        vheader.setDefaultSectionSize(size.height() + 2)
        vheader.setMinimumSectionSize(5)
        vheader.setSectionResizeMode(QHeaderView.Fixed)

        # Limit the number of rows displayed in the QTableView
        # (workaround for QTBUG-18490 / QTBUG-28631)
        maxrows = (2 ** 31 - 1) // (vheader.defaultSectionSize() + 2)
        if rowcount > maxrows:
            sliceproxy = TableSliceProxy(
                parent=view, rowSlice=slice(0, maxrows))
            sliceproxy.setSourceModel(datamodel)
            # First reset the view (without this the header view retains
            # it's state - at this point invalid/broken)
            view.setModel(None)
            view.setModel(sliceproxy)

        assert view.model().rowCount() <= maxrows
        assert vheader.sectionSize(0) > 1 or datamodel.rowCount() == 0

        # update the header (attribute names)
        self._update_variable_labels(view)

        selmodel = BlockSelectionModel(
            view.model(), parent=view, selectBlocks=not self.select_rows)
        view.setSelectionModel(selmodel)
        view.selectionFinished.connect(self.update_selection)

    #noinspection PyBroadException
    def set_corner_text(self, table, text):
        """Set table corner text."""
        # As this is an ugly hack, do everything in
        # try - except blocks, as it may stop working in newer Qt.
        # pylint: disable=broad-except
        if not hasattr(table, "btn") and not hasattr(table, "btnfailed"):
            try:
                btn = table.findChild(QAbstractButton)

                class Efc(QObject):
                    @staticmethod
                    def eventFilter(o, e):
                        if (isinstance(o, QAbstractButton) and
                                e.type() == QEvent.Paint):
                            # paint by hand (borrowed from QTableCornerButton)
                            btn = o
                            opt = QStyleOptionHeader()
                            opt.initFrom(btn)
                            state = QStyle.State_None
                            if btn.isEnabled():
                                state |= QStyle.State_Enabled
                            if btn.isActiveWindow():
                                state |= QStyle.State_Active
                            if btn.isDown():
                                state |= QStyle.State_Sunken
                            opt.state = state
                            opt.rect = btn.rect()
                            opt.text = btn.text()
                            opt.position = QStyleOptionHeader.OnlyOneSection
                            painter = QStylePainter(btn)
                            painter.drawControl(QStyle.CE_Header, opt)
                            return True     # eat event
                        return False
                table.efc = Efc()
                # disconnect default handler for clicks and connect a new one, which supports
                # both selection and deselection of all data
                btn.clicked.disconnect()
                btn.installEventFilter(table.efc)
                btn.clicked.connect(self._on_select_all)
                table.btn = btn

                if sys.platform == "darwin":
                    btn.setAttribute(Qt.WA_MacSmallSize)

            except Exception:
                table.btnfailed = True

        if hasattr(table, "btn"):
            try:
                btn = table.btn
                btn.setText(text)
                opt = QStyleOptionHeader()
                opt.text = btn.text()
                s = btn.style().sizeFromContents(
                    QStyle.CT_HeaderSection,
                    opt, QSize(),
                    btn)
                if s.isValid():
                    table.verticalHeader().setMinimumWidth(s.width())
            except Exception:
                pass

    def _set_input_summary(self, slot):
        def format_summary(summary):
            if isinstance(summary, ApproxSummary):
                length = summary.len.result() if summary.len.done() else \
                    summary.approx_len
            elif isinstance(summary, Summary):
                length = summary.len
            return length

        summary, details = self.info.NoInput, ""
        if slot:
            summary = format_summary(slot.summary)
            details = format_summary_details(slot.table)
        self.info.set_input_summary(summary, details)

        self.info_text.setText("\n".join(self._info_box_text(slot)))

    @staticmethod
    def _info_box_text(slot):
        def format_part(part):
            if isinstance(part, DenseArray):
                if not part.nans:
                    return ""
                perc = 100 * part.nans / (part.nans + part.non_nans)
                return f"\nmissing data - {int(part.nans)}"
                # return f" ({perc:.1f} % missing data, {part.nans} )"


            if isinstance(part, SparseArray):
                tag = "sparse"
            elif isinstance(part, SparseBoolArray):
                tag = "tags"
            else:  # isinstance(part, NotAvailable)
                return ""
            dens = 100 * part.non_nans / (part.nans + part.non_nans)
            return f" ({tag}, density {dens:.2f} %)"

        def desc(n, part):
            if n == 0:
                return f"No {part}s"
            elif n == 1:
                return f"1 {part}"
            else:
                return f"{n} {part}s"

        if slot is None:
            return ["No data."]
        summary = slot.summary
        text = []
        if isinstance(summary, ApproxSummary):
            if summary.len.done():
                # text.append(f"{summary.len.result()} instances")
                text.append("")
            else:
                # text.append(f"~{summary.approx_len} instances")
                text.append("")
        elif isinstance(summary, Summary):
            # text.append(f"{summary.len} instances")
            text.append("")
            if sum(p.nans for p in [summary.X, summary.Y, summary.M]) == 0:
                text[-1] += "(no missing data)"

        text.append(format_part(summary.X))
        # text.append(desc(len(summary.domain.attributes), "feature")
        #             + format_part(summary.X))

        if not summary.domain.class_vars:
            # text.append("No target variable.")
            text.append("")

        else:
            if len(summary.domain.class_vars) > 1:
                c_text = desc(len(summary.domain.class_vars), "outcome")
            elif summary.domain.has_continuous_class:
                c_text = "Numeric outcome"
            else:
                c_text = "Target with " \
                    + desc(len(summary.domain.class_var.values), "value")
            text.append(c_text + format_part(summary.Y))

        # text.append(desc(len(summary.domain.metas), "meta attribute")
                    # + format_part(summary.M))

        text.append("")

        # kh
        print("*********************************")
        print("********** owdatavalidation.py, FILE_PATH => ", owfile.FILE_PATH)
        print("*********************************")
      
        return text

    def _on_select_all(self, _):
        data_info = self.tabs.currentWidget().input_slot.summary
        if len(self.selected_rows) == data_info.len \
                and len(self.selected_cols) == len(data_info.domain.variables):
            self.tabs.currentWidget().clearSelection()
        else:
            self.tabs.currentWidget().selectAll()

    def _on_current_tab_changed(self, index):
        """Update the status bar on current tab change"""
        view = self.tabs.widget(index)
        if view is not None and view.model() is not None:
            self._set_input_summary(view.input_slot)
            self.update_selection()
        else:
            self._set_input_summary(None)

    def _update_variable_labels(self, view):
        "Update the variable labels visibility for `view`"
        model = view.model()
        if isinstance(model, TableSliceProxy):
            model = model.sourceModel()

        if self.show_attribute_labels:
            model.setRichHeaderFlags(
                RichTableModel.Labels | RichTableModel.Name)

            labelnames = set()
            domain = model.source.domain
            for a in itertools.chain(domain.metas, domain.variables):
                labelnames.update(a.attributes.keys())
            labelnames = sorted(
                [label for label in labelnames if not label.startswith("_")])
            self.set_corner_text(view, "\n".join([""] + labelnames))
        else:
            model.setRichHeaderFlags(RichTableModel.Name)
            self.set_corner_text(view, "")

    def _on_show_variable_labels_changed(self):
        """The variable labels (var.attribues) visibility was changed."""
        for slot in self._inputs:
            self._update_variable_labels(slot.view)

    def _on_distribution_color_changed(self):
        for ti in range(self.tabs.count()):
            widget = self.tabs.widget(ti)
            model = widget.model()
            while isinstance(model, QAbstractProxyModel):
                model = model.sourceModel()
            data = model.source
            class_var = data.domain.class_var
            if self.color_by_class and class_var and class_var.is_discrete:
                color_schema = [QColor(*c) for c in class_var.colors]
            else:
                color_schema = None
            if self.show_distributions:
                delegate = TableBarItemDelegate(widget, color=self.dist_color,
                                                color_schema=color_schema)
            else:
                delegate = TableDataDelegate(widget)
            widget.setItemDelegate(delegate)
        tab = self.tabs.currentWidget()
        if tab:
            tab.reset()

    def _mmm_stan_model(self):
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import pandas as pd
        import time
        from datetime import datetime
        from datetime import timedelta
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pystan
        import os

        with open("DataValidation.json", 'r') as f:
            vars = json.load(f)
        
        sns.color_palette("husl")
        sns.set_style('darkgrid')
        review_output_dir: str = "MMM_Stan_Model"
        if not os.path.isdir(review_output_dir):
            os.mkdir(review_output_dir)
        output_folder = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(review_output_dir, output_folder)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        df = pd.read_csv(owfile.FILE_PATH)
        df.to_csv(os.path.join(output_dir, "original_data.csv"), index=False)
        mdip_cols=[col for col in df.columns if vars["mdip_cols"] in col]
        mdsp_cols=[col for col in df.columns if vars["mdsp_cols"] in col]
        me_cols = [col for col in df.columns if vars["me_cols"] in col]
        st_cols = [vars["st_cols"]]
        mrkdn_cols = [col for col in df.columns if vars["mrkdn_cols"] in col]
        hldy_cols = [col for col in df.columns if vars["hldy_cols"] in col]
        seas_cols = [col for col in df.columns if vars["seas_cols"] in col]
        base_vars = me_cols+st_cols+mrkdn_cols+hldy_cols+seas_cols
        sales_cols =[vars["sales_cols"]]
        head = df[['wk_strt_dt']+mdip_cols+['sales']].head()
        head.to_csv(os.path.join(output_dir, "head.csv"), index=False)
        plt.figure(figsize=(24,20))
        sns.heatmap(df[mdip_cols+['sales']].corr(), square=True, annot=True, vmax=1, vmin=-1, cmap='RdBu')
        plt.savefig(os.path.join(output_dir, "first.png"))
        plt.figure(figsize=(50,50))
        sns.pairplot(df[mdip_cols+['sales']], vars=mdip_cols+['sales'])
        plt.savefig(os.path.join(output_dir, "second.png"))

        def apply_adstock(x, L, P, D):
            '''
            params:
            x: original media variable, array
            L: length
            P: peak, delay in effect
            D: decay, retain rate
            returns:
            array, adstocked media variable
            '''
            x = np.append(np.zeros(L-1), x)
            weights = np.zeros(L)
            for l in range(L):
                weight = D**((l-P)**2)
                weights[L-1-l] = weight
            adstocked_x = []
            for i in range(L-1, len(x)):
                x_array = x[i-L+1:i+1]
                xi = sum(x_array * weights)/sum(weights)
                adstocked_x.append(xi)
            adstocked_x = np.array(adstocked_x)
            return adstocked_x

        def adstock_transform(df, md_cols, adstock_params):
            '''
            params:
            df: original data
            md_cols: list, media variables to be transformed
            adstock_params: dict, 
                e.g., {'sem': {'L': 8, 'P': 0, 'D': 0.1}, 'dm': {'L': 4, 'P': 1, 'D': 0.7}}
            returns: 
            adstocked df
            '''
            md_df = pd.DataFrame()
            for md_col in md_cols:
                md = md_col.split('_')[-1]
                L, P, D = adstock_params[md]['L'], adstock_params[md]['P'], adstock_params[md]['D']
                xa = apply_adstock(df[md_col].values, L, P, D)
                md_df[md_col] = xa
            return md_df

        def hill_transform(x, ec, slope):
            return 1 / (1 + (x / ec)**(-slope))

        os.environ['CC'] = 'gcc-10'
        os.environ['CXX'] = 'g++-10'
        from sklearn.metrics import mean_squared_error

        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def apply_mean_center(x):
            mu = np.mean(x)
            xm = x/mu
            return xm, mu

        def mean_center_trandform(df, cols):
            '''
            returns: 
            mean-centered df
            scaler, dict
            '''
            df_new = pd.DataFrame()
            sc = {}
            for col in cols:
                x = df[col].values
                df_new[col], mu = apply_mean_center(x)
                sc[col] = mu
            return df_new, sc

        def mean_log1p_trandform(df, cols):
            '''
            returns: 
            mean-centered, log1p transformed df
            scaler, dict
            '''
            df_new = pd.DataFrame()
            sc = {}
            for col in cols:
                x = df[col].values
                xm, mu = apply_mean_center(x)
                sc[col] = mu
                df_new[col] = np.log1p(xm)
            return df_new, sc

        def save_json(data, file_name):
            with open(file_name, 'w') as fp:
                json.dump(data, fp)

        def load_json(file_name):
            with open(file_name, 'r') as fp:
                data = json.load(fp)
            return data

        df_ctrl, sc_ctrl = mean_center_trandform(df, ['sales']+me_cols+st_cols+mrkdn_cols)
        df_ctrl = pd.concat([df_ctrl, df[hldy_cols+seas_cols]], axis=1)
        pos_vars = [col for col in base_vars if col not in seas_cols]
        X1 = df_ctrl[pos_vars].values
        pn_vars = seas_cols
        X2 = df_ctrl[pn_vars].values

        ctrl_data = {
            'N': len(df_ctrl),
            'K1': len(pos_vars), 
            'K2': len(pn_vars), 
            'X1': X1,
            'X2': X2, 
            'y': df_ctrl['sales'].values,
            'max_intercept': min(df_ctrl['sales'])
        }

        ctrl_code1 = vars["ctrl_code1"]
        sm1 = pystan.StanModel(model_code=ctrl_code1, verbose=True)
        fit1 = sm1.sampling(data=ctrl_data, iter=2000, chains=4)
        fit1_result = fit1.extract()

        def extract_ctrl_model(fit_result, pos_vars=pos_vars, pn_vars=pn_vars, extract_param_list=False):
            ctrl_model = {}
            ctrl_model['pos_vars'] = pos_vars
            ctrl_model['pn_vars'] = pn_vars
            ctrl_model['beta1'] = fit_result['beta1'].mean(axis=0).tolist()
            ctrl_model['beta2'] = fit_result['beta2'].mean(axis=0).tolist()
            ctrl_model['alpha'] = fit_result['alpha'].mean()
            if extract_param_list:
                ctrl_model['beta1_list'] = fit_result['beta1'].tolist()
                ctrl_model['beta2_list'] = fit_result['beta2'].tolist()
                ctrl_model['alpha_list'] = fit_result['alpha'].tolist()
            return ctrl_model

        def ctrl_model_predict(ctrl_model, df):
            pos_vars, pn_vars = ctrl_model['pos_vars'], ctrl_model['pn_vars'] 
            X1, X2 = df[pos_vars], df[pn_vars]
            beta1, beta2 = np.array(ctrl_model['beta1']), np.array(ctrl_model['beta2'])
            alpha = ctrl_model['alpha']
            y_pred = np.dot(X1, beta1) + np.dot(X2, beta2) + alpha
            return y_pred

        base_sales_model = extract_ctrl_model(fit1_result, pos_vars=pos_vars, pn_vars=pn_vars)
        base_sales = ctrl_model_predict(base_sales_model, df_ctrl)
        df['base_sales'] = base_sales*sc_ctrl['sales']
        print('mape: ', mean_absolute_percentage_error(df['sales'], df['base_sales']))
        df_mmm, sc_mmm = mean_log1p_trandform(df, ['sales', 'base_sales'])
        mu_mdip = df[mdip_cols].apply(np.mean, axis=0).values
        max_lag = 8
        num_media = len(mdip_cols)
        X_media = np.concatenate((np.zeros((max_lag-1, num_media)), df[mdip_cols].values), axis=0)
        X_ctrl = df_mmm['base_sales'].values.reshape(len(df),1)

        model_data2 = {
            'N': len(df),
            'max_lag': max_lag, 
            'num_media': num_media,
            'X_media': X_media, 
            'mu_mdip': mu_mdip,
            'num_ctrl': X_ctrl.shape[1],
            'X_ctrl': X_ctrl, 
            'y': df_mmm['sales'].values
        }

        model_code2 = vars["model_code2"]

        sm2 = pystan.StanModel(model_code=model_code2, verbose=True)
        fit2 = sm2.sampling(data=model_data2, iter=1000, chains=3)
        fit2_result = fit2.extract()

        def extract_mmm(fit_result, max_lag=max_lag, media_vars=mdip_cols, ctrl_vars=['base_sales'], extract_param_list=True):
            mmm = {}
            mmm['max_lag'] = max_lag
            mmm['media_vars'], mmm['ctrl_vars'] = media_vars, ctrl_vars
            mmm['decay'] = decay = fit_result['decay'].mean(axis=0).tolist()
            mmm['peak'] = peak = fit_result['peak'].mean(axis=0).tolist()
            mmm['beta'] = fit_result['beta'].mean(axis=0).tolist()
            mmm['tau'] = fit_result['tau'].mean()
            if extract_param_list:
                mmm['decay_list'] = fit_result['decay'].tolist()
                mmm['peak_list'] = fit_result['peak'].tolist()
                mmm['beta_list'] = fit_result['beta'].tolist()
                mmm['tau_list'] = fit_result['tau'].tolist()
            adstock_params = {}
            media_names = [col.replace('mdip_', '') for col in media_vars]
            for i in range(len(media_names)):
                adstock_params[media_names[i]] = {
                    'L': max_lag,
                    'P': peak[i],
                    'D': decay[i]
                }
            mmm['adstock_params'] = adstock_params
            return mmm

        mmm = extract_mmm(fit2, max_lag=max_lag, media_vars=mdip_cols, ctrl_vars=['base_sales'])

        beta_media = {}
        for i in range(len(mmm['media_vars'])):
            md = mmm['media_vars'][i]
            betas = []
            for j in range(len(mmm['beta_list'])):
                betas.append(mmm['beta_list'][j][i])
            beta_media[md] = np.array(betas)
        f = plt.figure(figsize=(18,15))
        for i in range(len(mmm['media_vars'])):
            ax = f.add_subplot(5,3,i+1)
            md = mmm['media_vars'][i]
            x = beta_media[md]
            mean_x = x.mean()
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-')
            ax.axvline(median_x, color='g', linestyle='-')
            ax.set_title(md)
            plt.savefig(os.path.join(output_dir, "third.png"))

        def mmm_decompose_contrib(mmm, df, original_sales=df['sales']):
            adstock_params = mmm['adstock_params']
            beta, tau = mmm['beta'], mmm['tau']
            media_vars, ctrl_vars = mmm['media_vars'], mmm['ctrl_vars']
            num_media, num_ctrl = len(media_vars), len(ctrl_vars)
            X_media2 = adstock_transform(df, media_vars, adstock_params)
            X_media2, sc_mmm2 = mean_center_trandform(X_media2, media_vars)
            X_media2 = X_media2 + 1
            X_ctrl2, sc_mmm2_1 = mean_center_trandform(df[ctrl_vars], ctrl_vars)
            X_ctrl2 = X_ctrl2 + 1
            y_true2, sc_mmm2_2 = mean_center_trandform(df, ['sales'])
            y_true2 = y_true2 + 1
            sc_mmm2.update(sc_mmm2_1)
            sc_mmm2.update(sc_mmm2_2)
            X2 = pd.concat([X_media2, X_ctrl2], axis=1)
            factor_df = pd.DataFrame(columns=media_vars+ctrl_vars+['intercept'])
            for i in range(num_media):
                colname = media_vars[i]
                factor_df[colname] = X2[colname] ** beta[i]
            for i in range(num_ctrl):
                colname = ctrl_vars[i]
                factor_df[colname] = X2[colname] ** beta[num_media+i]
            factor_df['intercept'] = np.exp(tau)
            y_pred = factor_df.apply(np.prod, axis=1)
            factor_df['y_pred'], factor_df['y_true2'] = y_pred, y_true2
            factor_df['baseline'] = factor_df[['intercept']+ctrl_vars].apply(np.prod, axis=1)
            mc_df = pd.DataFrame(columns=media_vars+['baseline'])
            for col in media_vars:
                mc_df[col] = factor_df['y_true2'] - factor_df['y_true2']/factor_df[col]
            mc_df['baseline'] = factor_df['baseline']
            mc_df['y_true2'] = factor_df['y_true2']
            mc_df['mc_pred'] = mc_df[media_vars].apply(np.sum, axis=1)
            mc_df['mc_true'] = mc_df['y_true2'] - mc_df['baseline']
            mc_df['mc_delta'] =  mc_df['mc_pred'] - mc_df['mc_true']
            for col in media_vars:
                mc_df[col] = mc_df[col] - mc_df['mc_delta']*mc_df[col]/mc_df['mc_pred']
            mc_df['sales'] = original_sales
            for col in media_vars+['baseline']:
                mc_df[col] = mc_df[col]*mc_df['sales']/mc_df['y_true2']
            print('rmse (log-log model): ', 
                mean_squared_error(np.log(y_true2), np.log(y_pred)) ** (1/2))
            print('mape (multiplicative model): ', 
                mean_absolute_percentage_error(y_true2, y_pred))
            return mc_df

        def calc_media_contrib_pct(mc_df, media_vars=mdip_cols, sales_col='sales', period=52):
            '''
            returns:
            mc_pct: percentage over total sales
            mc_pct2: percentage over incremental sales (sales contributed by media channels)
            '''
            mc_pct = {}
            mc_pct2 = {}
            s = 0
            if period is None:
                for col in (media_vars+['baseline']):
                    mc_pct[col] = (mc_df[col]/mc_df[sales_col]).mean()
            else:
                for col in (media_vars+['baseline']):
                    mc_pct[col] = (mc_df[col]/mc_df[sales_col])[-period:].mean()
            for m in media_vars:
                s += mc_pct[m]
            for m in media_vars:
                mc_pct2[m] = mc_pct[m]/s
            return mc_pct, mc_pct2

        mc_df = mmm_decompose_contrib(mmm, df, original_sales=df['sales'])
        adstock_params = mmm['adstock_params']
        mc_pct, mc_pct2 = calc_media_contrib_pct(mc_df, period=52)

        def create_hill_model_data(df, mc_df, adstock_params, media):
            y = mc_df['mdip_'+media].values
            L, P, D = adstock_params[media]['L'], adstock_params[media]['P'], adstock_params[media]['D']
            x = df['mdsp_'+media].values
            x_adstocked = apply_adstock(x, L, P, D)
            # centralize
            mu_x, mu_y = x_adstocked.mean(), y.mean()
            sc = {'x': mu_x, 'y': mu_y}
            x = x_adstocked/mu_x
            y = y/mu_y
                
            model_data = {
                'N': len(y),
                'y': y,
                'X': x
            }
            return model_data, sc

        model_code3 = vars["model_code3"]

        def train_hill_model(df, mc_df, adstock_params, media, sm):
            '''
            params:
            df: original data
            mc_df: media contribution df derived from MMM
            adstock_params: adstock parameter dict output by MMM
            media: 'dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem'
            sm: stan model object    
            returns:
            a dict of model data, scaler, parameters
            '''
            data, sc = create_hill_model_data(df, mc_df, adstock_params, media)
            fit = sm.sampling(data=data, iter=2000, chains=4)
            fit_result = fit.extract()
            hill_model = {
                'beta_hill_list': fit_result['beta_hill'].tolist(),
                'ec_list': fit_result['ec'].tolist(),
                'slope_list': fit_result['slope'].tolist(),
                'sc': sc,
                'data': {
                    'X': data['X'].tolist(),
                    'y': data['y'].tolist(),
                }
            }
            return hill_model

        def extract_hill_model_params(hill_model, method='mean'):
            if method=='mean':
                hill_model_params = {
                    'beta_hill': np.mean(hill_model['beta_hill_list']), 
                    'ec': np.mean(hill_model['ec_list']), 
                    'slope': np.mean(hill_model['slope_list'])
                }
            elif method=='median':
                hill_model_params = {
                    'beta_hill': np.median(hill_model['beta_hill_list']), 
                    'ec': np.median(hill_model['ec_list']), 
                    'slope': np.median(hill_model['slope_list'])
                }
            return hill_model_params

        def hill_model_predict(hill_model_params, x):
            beta_hill, ec, slope = hill_model_params['beta_hill'], hill_model_params['ec'], hill_model_params['slope']
            y_pred = beta_hill * hill_transform(x, ec, slope)
            return y_pred

        def evaluate_hill_model(hill_model, hill_model_params):
            x = np.array(hill_model['data']['X'])
            y_true = np.array(hill_model['data']['y']) * hill_model['sc']['y']
            y_pred = hill_model_predict(hill_model_params, x) * hill_model['sc']['y']
            print('mape on original data: ', 
                mean_absolute_percentage_error(y_true, y_pred))
            return y_true, y_pred

        sm3 = pystan.StanModel(model_code=model_code3, verbose=True)
        hill_models = {}
        to_train = ['dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem']
        for media in to_train:
            print('training for media: ', media)
            hill_model = train_hill_model(df, mc_df, adstock_params, media, sm3)
            hill_models[media] = hill_model
        hill_model_params_mean, hill_model_params_med = {}, {}
        for md in list(hill_models.keys()):
            hill_model = hill_models[md]
            params1 = extract_hill_model_params(hill_model, method='mean')
            params1['sc'] = hill_model['sc']
            hill_model_params_mean[md] = params1
        for md in list(hill_models.keys()):
            print('evaluating media: ', md)
            hill_model = hill_models[md]
            hill_model_params = hill_model_params_mean[md]
            _ = evaluate_hill_model(hill_model, hill_model_params)
        f = plt.figure(figsize=(18,12))
        hm_keys = list(hill_models.keys())
        for i in range(len(hm_keys)):
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            x = hill_models[md]['ec_list']
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set_title(md)
            plt.savefig(os.path.join(output_dir, "fourth.png"))
        f = plt.figure(figsize=(18,12))
        hm_keys = list(hill_models.keys())
        for i in range(len(hm_keys)):
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            x = hill_models[md]['slope_list']
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set_title(md)
            plt.savefig(os.path.join(output_dir, "fifth.png"))
        f = plt.figure(figsize=(18,16))
        hm_keys = list(hill_models.keys())
        for i in range(len(hm_keys)):
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            hm = hill_models[md]
            hmp = hill_model_params_mean[md]
            x, y = hm['data']['X'], hm['data']['y']
            #mu_x, mu_y = hm['sc']['x'], hm['sc']['y']
            ec, slope = hmp['ec'], hmp['slope']
            x_sorted = np.array(sorted(x))
            y_fit = hill_model_predict(hmp, x_sorted)
            ax = sns.scatterplot(x=x, y=y, alpha=0.2)
            ax = sns.lineplot(x=x_sorted, y=y_fit, color='r', 
                        label='ec=%.2f, slope=%.2f'%(ec, slope))
            ax.set_title(md)
            plt.savefig(os.path.join(output_dir, "sixth.png"))
        ms_df = pd.DataFrame()
        for md in list(hill_models.keys()):
            hill_model = hill_models[md]
            x = np.array(hill_model['data']['X']) * hill_model['sc']['x']
            ms_df['mdsp_'+md] = x
        
        def calc_roas(mc_df, ms_df, period=None):
            roas = {}
            md_names = [col.split('_')[-1] for col in ms_df.columns]
            for i in range(len(md_names)):
                md = md_names[i]
                sp, mc = ms_df['mdsp_'+md], mc_df['mdip_'+md]
                if period is None:
                    md_roas = mc.sum()/sp.sum()
                else:
                    md_roas = mc[-period:].sum()/sp[-period:].sum()
                roas[md] = md_roas
            return roas

        def calc_weekly_roas(mc_df, ms_df):
            weekly_roas = pd.DataFrame()
            md_names = [col.split('_')[-1] for col in ms_df.columns]
            for md in md_names:
                weekly_roas[md] = mc_df['mdip_'+md]/ms_df['mdsp_'+md]
            weekly_roas.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            return weekly_roas

        roas_1y = calc_roas(mc_df, ms_df, period=52)
        weekly_roas = calc_weekly_roas(mc_df, ms_df)
        roas1y_df = pd.DataFrame(index=weekly_roas.columns.tolist())
        roas1y_df['roas_mean'] = weekly_roas[-52:].apply(np.mean, axis=0)
        roas1y_df['roas_median'] = weekly_roas[-52:].apply(np.median, axis=0)
        f = plt.figure(figsize=(18,12))
        for i in range(len(weekly_roas.columns)):
            md = weekly_roas.columns[i]
            ax = f.add_subplot(4,3,i+1)
            x = weekly_roas[md][-52:]
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set(xlabel=None)
            ax.set_title(md)
            plt.savefig(os.path.join(output_dir, "seventh.png"))
        
        def calc_mroas(hill_model, hill_model_params, period=52):
            '''
            calculate mROAS for a media
            params:
            hill_model: a dict containing model data and scaling factor
            hill_model_params: a dict containing beta_hill, ec, slope
            period: in weeks, the period used to calculate ROAS and mROAS. 52 is last one year.
            return:
            mROAS value
            '''
            mu_x, mu_y = hill_model['sc']['x'], hill_model['sc']['y']
            # get current media spending level over the period specified
            cur_sp = np.asarray(hill_model['data']['X'])
            if period is not None:
                cur_sp = cur_sp[-period:]
            cur_mc = sum(hill_model_predict(hill_model_params, cur_sp) * mu_y)
            # next spending level: increase by 1%
            next_sp = cur_sp * 1.01
            # media contribution under next spending level
            next_mc = sum(hill_model_predict(hill_model_params, next_sp) * mu_y)
            
            # mROAS
            delta_mc = next_mc - cur_mc
            delta_sp = sum(next_sp * mu_x) - sum(cur_sp * mu_x)
            mroas = delta_mc/delta_sp
            return mroas

        mroas_1y = {}
        for md in list(hill_models.keys()):
            hill_model = hill_models[md]
            hill_model_params = hill_model_params_mean[md]
            mroas_1y[md] = calc_mroas(hill_model, hill_model_params, period=52)
        roas1y_df = pd.concat([
            roas1y_df[['roas_mean', 'roas_median']],
            pd.DataFrame.from_dict(mroas_1y, orient='index', columns=['mroas']),
            pd.DataFrame.from_dict(roas_1y, orient='index', columns=['roas_avg'])
        ], axis=1)
        roas1y_df.to_csv(os.path.join(output_dir, "roasly_df.csv"), index=False)
        QMessageBox.about(self, "Data Validation", "Success")

    def _on_select_rows_changed(self):
        for slot in self._inputs:
            selection_model = slot.view.selectionModel()
            selection_model.setSelectBlocks(not self.select_rows)
            if self.select_rows:
                slot.view.setSelectionBehavior(QTableView.SelectRows)
                # Expand the current selection to full row selection.
                selection_model.select(
                    selection_model.selection(),
                    QItemSelectionModel.Select | QItemSelectionModel.Rows
                )
            else:
                slot.view.setSelectionBehavior(QTableView.SelectItems)

    def restore_order(self):
        """Restore the original data order of the current view."""
        table = self.tabs.currentWidget()
        if table is not None:
            table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    @Slot()
    def _update_info(self):
        current = self.tabs.currentWidget()
        if current is not None and current.model() is not None:
            self._set_input_summary(current.input_slot)

    def update_selection(self, *_):
        self.commit.deferred()

    def set_selection(self):
        if self.selected_rows and self.selected_cols:
            view = self.tabs.currentWidget()
            model = view.model()
            if model.rowCount() <= self.selected_rows[-1] or \
                    model.columnCount() <= self.selected_cols[-1]:
                return

            selection = QItemSelection()
            rowranges = list(ranges(self.selected_rows))
            colranges = list(ranges(self.selected_cols))

            for rowstart, rowend in rowranges:
                for colstart, colend in colranges:
                    selection.append(
                        QItemSelectionRange(
                            view.model().index(rowstart, colstart),
                            view.model().index(rowend - 1, colend - 1)
                        )
                    )
            view.selectionModel().select(
                selection, QItemSelectionModel.ClearAndSelect)

    @staticmethod
    def get_selection(view):
        """
        Return the selected row and column indices of the selection in view.
        """
        selmodel = view.selectionModel()

        selection = selmodel.selection()
        model = view.model()
        # map through the proxies into input table.
        while isinstance(model, QAbstractProxyModel):
            selection = model.mapSelectionToSource(selection)
            model = model.sourceModel()

        assert isinstance(selmodel, BlockSelectionModel)
        assert isinstance(model, TableModel)

        row_spans, col_spans = selection_blocks(selection)
        rows = list(itertools.chain.from_iterable(itertools.starmap(range, row_spans)))
        cols = list(itertools.chain.from_iterable(itertools.starmap(range, col_spans)))
        rows = numpy.array(rows, dtype=numpy.intp)
        # map the rows through the applied sorting (if any)
        rows = model.mapToSourceRows(rows)
        rows = rows.tolist()
        return rows, cols

    @staticmethod
    def _get_model(view):
        model = view.model()
        while isinstance(model, QAbstractProxyModel):
            model = model.sourceModel()
        return model

    @gui.deferred
    def commit(self):
        """
        Commit/send the current selected row/column selection.
        """
        selected_data = table = rowsel = None
        view = self.tabs.currentWidget()
        if view and view.model() is not None:
            model = self._get_model(view)
            table = model.source  # The input data table

            # Selections of individual instances are not implemented
            # for SqlTables
            if isinstance(table, SqlTable):
                self.Outputs.selected_data.send(selected_data)
                self.Outputs.annotated_data.send(None)
                return

            rowsel, colsel = self.get_selection(view)
            self.selected_rows, self.selected_cols = rowsel, colsel

            domain = table.domain

            if len(colsel) < len(domain.variables) + len(domain.metas):
                # only a subset of the columns is selected
                allvars = domain.class_vars + domain.metas + domain.attributes
                columns = [(c, model.headerData(c, Qt.Horizontal,
                                                TableModel.DomainRole))
                           for c in colsel]
                assert all(role is not None for _, role in columns)

                def select_vars(role):
                    """select variables for role (TableModel.DomainRole)"""
                    return [allvars[c] for c, r in columns if r == role]

                attrs = select_vars(TableModel.Attribute)
                if attrs and issparse(table.X):
                    # for sparse data you can only select all attributes
                    attrs = table.domain.attributes
                class_vars = select_vars(TableModel.ClassVar)
                metas = select_vars(TableModel.Meta)
                domain = Orange.data.Domain(attrs, class_vars, metas)

            # Send all data by default
            if not rowsel:
                selected_data = table
            else:
                selected_data = table.from_table(domain, table, rowsel)

        self.Outputs.selected_data.send(selected_data)
        self.Outputs.annotated_data.send(create_annotated_table(table, rowsel))

    def copy(self):
        """
        Copy current table selection to the clipboard.
        """
        view = self.tabs.currentWidget()
        if view is not None:
            mime = table_selection_to_mime_data(view)
            QApplication.clipboard().setMimeData(
                mime, QClipboard.Clipboard
            )

    def send_report(self):
        view = self.tabs.currentWidget()
        if not view or not view.model():
            return
        model = self._get_model(view)
        self.report_data_brief(model.source)
        self.report_table(view)


# Table Summary

# Basic statistics for X/Y/metas arrays
DenseArray = namedtuple(
    "DenseArray", ["nans", "non_nans", "stats"])
SparseArray = namedtuple(
    "SparseArray", ["nans", "non_nans", "stats"])
SparseBoolArray = namedtuple(
    "SparseBoolArray", ["nans", "non_nans", "stats"])
NotAvailable = namedtuple("NotAvailable", [])

#: Orange.data.Table summary
Summary = namedtuple(
    "Summary",
    ["len", "domain", "X", "Y", "M"])

#: Orange.data.sql.table.SqlTable summary
ApproxSummary = namedtuple(
    "ApproxSummary",
    ["approx_len", "len", "domain", "X", "Y", "M"])


def table_summary(table):
    if isinstance(table, SqlTable):
        approx_len = table.approx_len()
        len_future = concurrent.futures.Future()

        def _len():
            len_future.set_result(len(table))
        threading.Thread(target=_len).start()  # KILL ME !!!

        return ApproxSummary(approx_len, len_future, table.domain,
                             NotAvailable(), NotAvailable(), NotAvailable())
    else:
        domain = table.domain
        n_instances = len(table)
        # dist = basic_stats.DomainBasicStats(table, include_metas=True)
        bstats = datacaching.getCached(
            table, basic_stats.DomainBasicStats, (table, True)
        )

        dist = bstats.stats
        # pylint: disable=unbalanced-tuple-unpacking
        X_dist, Y_dist, M_dist = numpy.split(
            dist, numpy.cumsum([len(domain.attributes),
                                len(domain.class_vars)]))

        def parts(array, density, col_dist):
            array = numpy.atleast_2d(array)
            nans = sum([dist.nans for dist in col_dist])
            non_nans = sum([dist.non_nans for dist in col_dist])
            if density == Storage.DENSE:
                return DenseArray(nans, non_nans, col_dist)
            elif density == Storage.SPARSE:
                return SparseArray(nans, non_nans, col_dist)
            elif density == Storage.SPARSE_BOOL:
                return SparseBoolArray(nans, non_nans, col_dist)
            elif density == Storage.MISSING:
                return NotAvailable()
            else:
                assert False
                return None

        X_part = parts(table.X, table.X_density(), X_dist)
        Y_part = parts(table.Y, table.Y_density(), Y_dist)
        M_part = parts(table.metas, table.metas_density(), M_dist)
        return Summary(n_instances, domain, X_part, Y_part, M_part)


def is_sortable(table):
    if isinstance(table, SqlTable):
        return False
    elif isinstance(table, Orange.data.Table):
        return True
    else:
        return False


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWDataTable).run(
        insert_dataset=[
            (0, Table("iris")),
            (1, Table("brown-selected")),
            (2, Table("housing"))
        ])
